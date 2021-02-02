mod loader;
use crate::loader::*;

mod accel;
use crate::accel::*;

use bytemuck::{Pod, Zeroable};
use caldera::*;
use imgui::im_str;
use imgui::Key;
use spark::vk;
use std::sync::{Arc, Mutex};
use std::{env, mem, slice};
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct RasterData {
    proj_from_world: Mat4,
}

descriptor_set_layout!(RasterDescriptorSetLayout {
    test: UniformData<RasterData>,
});

descriptor_set_layout!(CopyDescriptorSetLayout { ids: StorageImage });

#[derive(Clone, Copy, Eq, PartialEq)]
enum RenderMode {
    Raster,
    RasterMultisampled,
    RayTrace,
}

struct App {
    context: Arc<Context>,

    raster_descriptor_set_layout: RasterDescriptorSetLayout,
    raster_pipeline_layout: vk::PipelineLayout,

    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    mesh_info: Arc<Mutex<MeshInfo>>,
    accel_info: Option<AccelInfo>,
    render_mode: RenderMode,
    is_rotating: bool,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: String) -> Self {
        let context = &base.context;
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let raster_descriptor_set_layout = RasterDescriptorSetLayout::new(descriptor_set_layout_cache);
        let raster_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(raster_descriptor_set_layout.0);

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(descriptor_set_layout_cache);
        let copy_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(copy_descriptor_set_layout.0);

        let mesh_info = Arc::new(Mutex::new(MeshInfo::new(&mut base.systems.resource_loader)));
        base.systems.resource_loader.async_load({
            let mesh_info = Arc::clone(&mesh_info);
            let with_ray_tracing = context.device.extensions.supports_khr_acceleration_structure();
            move |allocator| {
                let mut mesh_info_clone = *mesh_info.lock().unwrap();
                mesh_info_clone.load(allocator, &mesh_file_name, with_ray_tracing);
                *mesh_info.lock().unwrap() = mesh_info_clone;
            }
        });

        Self {
            context: Arc::clone(&context),
            raster_descriptor_set_layout,
            raster_pipeline_layout,
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            mesh_info,
            accel_info: None,
            render_mode: RenderMode::Raster,
            is_rotating: false,
            angle: PI / 8.0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(ui.key_index(Key::Escape)) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                let ui = &ui;
                let context = &base.context;
                let render_mode = &mut self.render_mode;
                let is_rotating = &mut self.is_rotating;
                move || {
                    ui.text("Render Mode:");
                    ui.radio_button(im_str!("Raster"), render_mode, RenderMode::Raster);
                    ui.radio_button(
                        im_str!("Raster (Multisampled)"),
                        render_mode,
                        RenderMode::RasterMultisampled,
                    );
                    if context.device.extensions.supports_khr_acceleration_structure() {
                        ui.radio_button(im_str!("Ray Trace"), render_mode, RenderMode::RayTrace);
                    } else {
                        ui.text_disabled("Ray Tracing Not Supported!");
                    }
                    ui.checkbox(im_str!("Rotate"), is_rotating);
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore);
        let swap_extent = base.display.swapchain.get_extent();
        let swap_format = base.display.swapchain.get_format();
        let swap_image = schedule.import_image(
            &ImageDesc::new_2d(
                swap_extent.width,
                swap_extent.height,
                swap_format,
                vk::ImageAspectFlags::COLOR,
            ),
            ImageUsage::COLOR_ATTACHMENT_WRITE | ImageUsage::SWAPCHAIN,
            swap_vk_image,
            ImageUsage::empty(),
        );

        let main_sample_count = if matches!(self.render_mode, RenderMode::RasterMultisampled) {
            vk::SampleCountFlags::N4
        } else {
            vk::SampleCountFlags::N1
        };
        let depth_image_desc = ImageDesc::new_2d(
            swap_extent.width,
            swap_extent.height,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        )
        .with_samples(main_sample_count);
        let depth_image = schedule.describe_image(&depth_image_desc);
        let mut main_render_state =
            RenderState::new(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]).with_depth_temp(depth_image);
        if main_sample_count != vk::SampleCountFlags::N1 {
            let msaa_image = schedule.describe_image(
                &ImageDesc::new_2d(
                    swap_extent.width,
                    swap_extent.height,
                    swap_format,
                    vk::ImageAspectFlags::COLOR,
                )
                .with_samples(main_sample_count),
            );
            main_render_state = main_render_state.with_color_temp(msaa_image);
        }

        let mesh_info = *self.mesh_info.lock().unwrap();
        let position_buffer = base.systems.resource_loader.get_buffer(mesh_info.position_buffer);
        let attribute_buffer = base.systems.resource_loader.get_buffer(mesh_info.attribute_buffer);
        let index_buffer = base.systems.resource_loader.get_buffer(mesh_info.index_buffer);
        let instance_buffer = base.systems.resource_loader.get_buffer(mesh_info.instance_buffer);

        let view_from_world = Isometry3::new(
            Vec3::new(0.0, 0.0, -6.0),
            Rotor3::from_rotation_yz(0.5) * Rotor3::from_rotation_xz(self.angle),
        );

        let vertical_fov = PI / 7.0;
        let aspect_ratio = (swap_extent.width as f32) / (swap_extent.height as f32);
        let proj_from_view = projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);

        if base.context.device.extensions.supports_khr_acceleration_structure()
            && self.accel_info.is_none()
            && position_buffer.is_some()
            && attribute_buffer.is_some()
            && index_buffer.is_some()
        {
            self.accel_info = Some(AccelInfo::new(
                &base.context,
                &mut base.systems.descriptor_set_layout_cache,
                &base.systems.pipeline_cache,
                &mut base.systems.resource_loader,
                &mesh_info,
                &mut base.systems.global_allocator,
                &mut schedule,
            ));
        }
        if let Some(accel_info) = self.accel_info.as_mut() {
            accel_info.update(
                &base.context,
                &base.systems.resource_loader,
                &mut base.systems.global_allocator,
                &mut schedule,
            );
        }
        let trace_image = if let Some(accel_info) = self
            .accel_info
            .as_ref()
            .filter(|_| matches!(self.render_mode, RenderMode::RayTrace))
        {
            let world_from_view = view_from_world.inversed();

            let xy_from_st =
                Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * vertical_fov).tan(), Vec2::zero());
            let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, 2.0), Vec2::new(1.0, -1.0));
            let coord_from_uv = Scale2Offset2::new(
                UVec2::new(swap_extent.width, swap_extent.height).as_float(),
                Vec2::zero(),
            );
            let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

            let ray_origin = world_from_view.translation;
            let ray_vec_from_coord = world_from_view.rotation.into_matrix()
                * Mat3::from_scale(-1.0)
                * xy_from_coord.into_homogeneous_matrix();

            accel_info.dispatch(
                &base.context,
                &base.systems.resource_loader,
                &mut schedule,
                &base.systems.descriptor_pool,
                &swap_extent,
                ray_origin,
                ray_vec_from_coord,
            )
        } else {
            None
        };

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if let Some(trace_image) = trace_image {
                    params.add_image(trace_image, ImageUsage::FRAGMENT_STORAGE_READ);
                }
            },
            {
                let context = &base.context;
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let copy_descriptor_set_layout = &self.copy_descriptor_set_layout;
                let copy_pipeline_layout = self.copy_pipeline_layout;
                let raster_descriptor_set_layout = &self.raster_descriptor_set_layout;
                let raster_pipeline_layout = self.raster_pipeline_layout;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_extent);

                    if let Some(trace_image) = trace_image {
                        let trace_image_view = params.get_image_view(trace_image);

                        let copy_descriptor_set = copy_descriptor_set_layout.write(&descriptor_pool, trace_image_view);

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count);

                        draw_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            copy_pipeline_layout,
                            &state,
                            "mesh/copy.vert.spv",
                            "mesh/copy.frag.spv",
                            copy_descriptor_set,
                            3,
                        );
                    } else if let (
                        Some(position_buffer),
                        Some(attribute_buffer),
                        Some(index_buffer),
                        Some(instance_buffer),
                    ) = (position_buffer, attribute_buffer, index_buffer, instance_buffer)
                    {
                        let raster_descriptor_set = raster_descriptor_set_layout.write(&descriptor_pool, |buf| {
                            *buf = RasterData {
                                proj_from_world: proj_from_view * view_from_world.into_homogeneous_matrix(),
                            };
                        });

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count).with_vertex_inputs(
                            &[
                                vk::VertexInputBindingDescription {
                                    binding: 0,
                                    stride: mem::size_of::<PositionData>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                vk::VertexInputBindingDescription {
                                    binding: 1,
                                    stride: mem::size_of::<AttributeData>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                vk::VertexInputBindingDescription {
                                    binding: 2,
                                    stride: mem::size_of::<InstanceData>() as u32,
                                    input_rate: vk::VertexInputRate::INSTANCE,
                                },
                            ],
                            &[
                                vk::VertexInputAttributeDescription {
                                    location: 0,
                                    binding: 0,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 1,
                                    binding: 1,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 2,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 3,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 16,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 4,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 32,
                                },
                            ],
                        );
                        let pipeline = pipeline_cache.get_graphics(
                            "mesh/raster.vert.spv",
                            "mesh/raster.frag.spv",
                            raster_pipeline_layout,
                            &state,
                        );
                        unsafe {
                            context
                                .device
                                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                            context.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                raster_pipeline_layout,
                                0,
                                slice::from_ref(&raster_descriptor_set),
                                &[],
                            );
                            context.device.cmd_bind_vertex_buffers(
                                cmd,
                                0,
                                &[position_buffer, attribute_buffer, instance_buffer],
                                &[0, 0, 0],
                            );
                            context
                                .device
                                .cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
                            context.device.cmd_draw_indexed(
                                cmd,
                                mesh_info.triangle_count * 3,
                                MeshInfo::INSTANCE_COUNT as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }

                    // draw imgui
                    ui_platform.prepare_render(&ui, window);

                    let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                    ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
                }
            },
        );

        schedule.run(
            &base.context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            swap_image,
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display.present(swap_vk_image, rendering_finished_semaphore);

        if self.is_rotating {
            self.angle += base.ui_context.io().delta_time;
        }
    }
}

fn main() {
    let mut params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        allow_ray_tracing: true,
        ..Default::default()
    };
    let mut mesh_file_name = None;
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        match arg {
            "--no-rays" => params.allow_ray_tracing = false,
            _ => {
                if !params.parse_arg(arg) {
                    if mesh_file_name.is_none() {
                        mesh_file_name = Some(arg.to_owned());
                    } else {
                        panic!("unknown argument {:?}", arg);
                    }
                }
            }
        }
    }
    let mesh_file_name = mesh_file_name.expect("missing PLY mesh filename argument");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("mesh")
        .with_inner_size(Size::Logical(LogicalSize::new(1920.0, 1080.0)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &params);
    let app = App::new(&mut base, mesh_file_name);

    let mut apps = Some((base, app));
    event_loop.run(move |event, target, control_flow| {
        match apps
            .as_mut()
            .map(|(base, _)| base)
            .unwrap()
            .process_event(&event, target, control_flow)
        {
            AppEventResult::None => {}
            AppEventResult::Redraw => {
                let (base, app) = apps.as_mut().unwrap();
                app.render(base);
            }
            AppEventResult::Destroy => {
                apps.take();
            }
        }
    });
}
