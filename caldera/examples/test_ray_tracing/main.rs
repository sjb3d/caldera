mod accel;
mod loader;

use crate::{accel::*, loader::*};
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use spark::vk;
use std::{mem, path::PathBuf, slice};
use structopt::StructOpt;
use strum::VariantNames;
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

descriptor_set!(RasterDescriptorSet {
    test: UniformData<RasterData>,
});

descriptor_set!(CopyDescriptorSet { ids: StorageImage });

#[derive(Clone, Copy, Eq, PartialEq)]
enum RenderMode {
    Raster,
    RasterMultisampled,
    RayTrace,
}

struct LoadResult {
    mesh_info: MeshInfo,
    accel_info: Option<AccelInfo>,
}

struct App {
    context: SharedContext,

    load_result: TaskOutput<LoadResult>,
    render_mode: RenderMode,
    is_rotating: bool,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: PathBuf) -> Self {
        let context = SharedContext::clone(&base.context);

        let has_ray_tracing = context
            .physical_device_features
            .ray_tracing_pipeline
            .ray_tracing_pipeline
            .as_bool();
        let resource_loader = base.systems.resource_loader.clone();
        let load_result = base.systems.task_system.spawn_task(async move {
            let mesh_info = MeshInfo::load(resource_loader.clone(), &mesh_file_name, has_ray_tracing).await;
            let accel_info = if has_ray_tracing {
                Some(AccelInfo::new(resource_loader.clone(), &mesh_info).await)
            } else {
                None
            };
            LoadResult { mesh_info, accel_info }
        });

        Self {
            context,
            load_result,
            render_mode: if has_ray_tracing {
                RenderMode::RayTrace
            } else {
                RenderMode::Raster
            },
            is_rotating: false,
            angle: PI / 8.0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let cbar = base.systems.acquire_command_buffer();

        base.ui_begin_frame();
        base.egui_ctx.clone().input(|i| {
            if i.key_pressed(egui::Key::Escape) {
                base.exit_requested = true;
            }
        });
        egui::Window::new("Debug")
            .default_pos([5.0, 5.0])
            .default_size([350.0, 150.0])
            .show(&base.egui_ctx, |ui| {
                ui.checkbox(&mut self.is_rotating, "Rotate");
                ui.label("Render Mode:");
                ui.radio_value(&mut self.render_mode, RenderMode::Raster, "Raster");
                ui.radio_value(
                    &mut self.render_mode,
                    RenderMode::RasterMultisampled,
                    "Raster (Multisampled)",
                );
                if self
                    .context
                    .physical_device_features
                    .ray_tracing_pipeline
                    .ray_tracing_pipeline
                    .as_bool()
                {
                    ui.radio_value(&mut self.render_mode, RenderMode::RayTrace, "Ray Trace");
                } else {
                    ui.label("Ray Tracing Not Supported!");
                }
            });
        base.systems.draw_ui(&base.egui_ctx);
        base.ui_end_frame(cbar.pre_swapchain_cmd);

        let mut schedule = base.systems.resource_loader.begin_schedule(
            &mut base.systems.render_graph,
            base.context.as_ref(),
            &base.systems.descriptor_pool,
            &base.systems.pipeline_cache,
        );

        let swap_vk_image = base
            .display
            .acquire(&base.window, cbar.image_available_semaphore.unwrap());
        let swap_size = base.display.swapchain.get_size();
        let swap_format = base.display.swapchain.get_format();
        let swap_image = schedule.import_image(
            &ImageDesc::new_2d(swap_size, swap_format, vk::ImageAspectFlags::COLOR),
            ImageUsage::COLOR_ATTACHMENT_WRITE | ImageUsage::SWAPCHAIN,
            swap_vk_image,
            ImageUsage::empty(),
            ImageUsage::SWAPCHAIN,
        );

        let main_sample_count = if matches!(self.render_mode, RenderMode::RasterMultisampled) {
            vk::SampleCountFlags::N4
        } else {
            vk::SampleCountFlags::N1
        };
        let depth_image_desc = ImageDesc::new_2d(swap_size, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH)
            .with_samples(main_sample_count);
        let depth_image = schedule.describe_image(&depth_image_desc);
        let mut main_render_state = RenderState::new()
            .with_color(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32])
            .with_depth(depth_image, AttachmentLoadOp::Clear, AttachmentStoreOp::None);
        if main_sample_count != vk::SampleCountFlags::N1 {
            let msaa_image = schedule.describe_image(
                &ImageDesc::new_2d(swap_size, swap_format, vk::ImageAspectFlags::COLOR).with_samples(main_sample_count),
            );
            main_render_state = main_render_state.with_color_temp(msaa_image);
        }

        let load_result = self.load_result.get();

        let view_from_world = Isometry3::new(
            Vec3::new(0.0, 0.0, -6.0),
            Rotor3::from_rotation_yz(0.5) * Rotor3::from_rotation_xz(self.angle),
        );

        let vertical_fov = PI / 7.0;
        let aspect_ratio = (swap_size.x as f32) / (swap_size.y as f32);
        let proj_from_view = projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);

        let mut trace_image = None;
        if let Some(load_result) = load_result {
            if let Some(accel_info) = load_result.accel_info.as_ref() {
                if matches!(self.render_mode, RenderMode::RayTrace) {
                    let world_from_view = view_from_world.inversed();

                    let xy_from_st =
                        Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * vertical_fov).tan(), Vec2::zero());
                    let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, 2.0), Vec2::new(1.0, -1.0));
                    let coord_from_uv = Scale2Offset2::new(swap_size.as_float(), Vec2::zero());
                    let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

                    let ray_origin = world_from_view.translation;
                    let ray_vec_from_coord = world_from_view.rotation.into_matrix()
                        * Mat3::from_scale(-1.0)
                        * xy_from_coord.into_homogeneous_matrix();

                    trace_image = Some(accel_info.dispatch(
                        &base.context,
                        &mut schedule,
                        &base.systems.descriptor_pool,
                        swap_size,
                        ray_origin,
                        ray_vec_from_coord,
                    ))
                }
            }
        }

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if let Some(trace_image) = trace_image {
                    params.add_image(trace_image, ImageUsage::FRAGMENT_STORAGE_READ);
                }
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let pixels_per_point = base.egui_ctx.pixels_per_point();
                let egui_renderer = &mut base.egui_renderer;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);

                    if let Some(trace_image) = trace_image {
                        let copy_descriptor_set = CopyDescriptorSet::create(
                            descriptor_pool,
                            params.get_image_view(trace_image, ImageViewDesc::default()),
                        );

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count);

                        draw_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            &state,
                            "test_ray_tracing/copy.vert.spv",
                            "test_ray_tracing/copy.frag.spv",
                            copy_descriptor_set,
                            3,
                        );
                    } else if let Some(mesh_info) = load_result.map(|load_result| &load_result.mesh_info) {
                        let raster_descriptor_set = RasterDescriptorSet::create(descriptor_pool, |buf| {
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
                        let raster_pipeline_layout =
                            pipeline_cache.get_pipeline_layout(slice::from_ref(&raster_descriptor_set.layout));
                        let pipeline = pipeline_cache.get_graphics(
                            VertexShaderDesc::standard("test_ray_tracing/raster.vert.spv"),
                            "test_ray_tracing/raster.frag.spv",
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
                                slice::from_ref(&raster_descriptor_set.set),
                                &[],
                            );
                            context.device.cmd_bind_vertex_buffers(
                                cmd,
                                0,
                                &[
                                    mesh_info.position_buffer,
                                    mesh_info.attribute_buffer,
                                    mesh_info.instance_buffer,
                                ],
                                &[0, 0, 0],
                            );
                            context.device.cmd_bind_index_buffer(
                                cmd,
                                Some(mesh_info.index_buffer),
                                0,
                                vk::IndexType::UINT32,
                            );
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

                    // draw ui
                    let egui_pipeline = pipeline_cache.get_ui(egui_renderer, render_pass, main_sample_count);
                    egui_renderer.render(
                        &context.device,
                        cmd,
                        egui_pipeline,
                        swap_size.x,
                        swap_size.y,
                        pixels_per_point,
                    );
                }
            },
        );

        schedule.run(
            &base.context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            Some(swap_image),
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display
            .present(swap_vk_image, rendering_finished_semaphore.unwrap());

        if self.is_rotating {
            self.angle += base.egui_ctx.input(|i| i.stable_dt);
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    /// Core Vulkan version to load
    #[structopt(short, long, parse(try_from_str=try_version_from_str), default_value="1.1")]
    version: vk::Version,

    /// Whether to use EXT_inline_uniform_block
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional")]
    inline_uniform_block: ContextFeature,

    /// Whether to use KHR_ray_tracing_pipeline
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional")]
    ray_tracing: ContextFeature,

    /// The PLY file to load
    mesh_file_name: PathBuf,
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        ray_tracing: app_params.ray_tracing,
        ..Default::default()
    };

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("mesh")
        .with_inner_size(Size::Logical(LogicalSize::new(1920.0, 1080.0)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &context_params);
    let app = App::new(&mut base, app_params.mesh_file_name);

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
