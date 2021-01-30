mod accel;
mod scene;

use crate::accel::*;
use crate::scene::*;
use caldera::*;
use caldera_macro::descriptor_set_layout;
use imgui::im_str;
use imgui::{Key, MouseButton};
use spark::vk;
use std::env;
use std::sync::Arc;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

descriptor_set_layout!(CopyDescriptorSetLayout { ids: StorageImage });

struct ViewDrag {
    pub rotation: Rotor3,
    drag_start: Option<(Rotor3, Vec3)>,
}

impl ViewDrag {
    fn new(rotation: Rotor3) -> Self {
        Self {
            rotation,
            drag_start: None,
        }
    }

    fn update(&mut self, io: &imgui::Io, dir_from_coord: impl FnOnce(Vec2) -> Vec3) {
        if io.want_capture_mouse {
            self.drag_start = None;
        } else {
            let dir_now = dir_from_coord(Vec2::from(io.mouse_pos));
            if io[MouseButton::Left] {
                if let Some((rotation_start, dir_start)) = self.drag_start {
                    self.rotation = rotation_start * Rotor3::from_rotation_between(dir_now, dir_start);
                } else {
                    self.drag_start = Some((self.rotation, dir_now));
                }
            } else {
                if let Some((rotation_start, dir_start)) = self.drag_start.take() {
                    self.rotation = rotation_start * Rotor3::from_rotation_between(dir_now, dir_start);
                }
            }
        }
    }
}

struct App {
    context: Arc<Context>,

    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    accel: SceneAccel,
    camera_ref: CameraRef,
    view_drag: ViewDrag,

    light: Option<QuadLight>,
}

impl App {
    fn new(base: &mut AppBase) -> Self {
        let context = &base.context;
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(descriptor_set_layout_cache);
        let copy_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(copy_descriptor_set_layout.0);

        let scene = create_cornell_box_scene();
        let accel = SceneAccel::new(
            scene,
            &context,
            &mut base.systems.descriptor_set_layout_cache,
            &base.systems.pipeline_cache,
            &mut base.systems.resource_loader,
        );

        let scene = accel.scene();
        let camera_ref = CameraRef(0);
        let rotation = scene.transform(scene.camera(camera_ref).transform_ref).0.rotation;

        let light = scene
            .instances
            .iter()
            .filter(|instance| scene.shader(instance.shader_ref).is_emissive())
            .filter_map(|instance| match *scene.geometry(instance.geometry_ref) {
                Geometry::TriangleMesh { .. } => None,
                Geometry::Quad { size, transform } => Some(QuadLight {
                    transform: scene.transform(instance.transform_ref).0 * transform,
                    size,
                    emission: scene.shader(instance.shader_ref).emission,
                }),
            })
            .next();

        Self {
            context: Arc::clone(&context),
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            accel,
            camera_ref,
            view_drag: ViewDrag::new(rotation),
            light,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        // TODO: move to an update function
        let io = base.ui_context.io();
        let scene = self.accel.scene();
        let camera = &scene.cameras[0];

        let display_size: Vec2 = io.display_size.into();
        let aspect_ratio = (display_size.x as f32) / (display_size.y as f32);

        let xy_from_st = Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * camera.fov_y).tan(), Vec2::zero());
        let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, -2.0), Vec2::new(1.0, 1.0));
        let coord_from_uv = Scale2Offset2::new(display_size, Vec2::zero());
        let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

        self.view_drag.update(base.ui_context.io(), |p: Vec2| {
            let v = xy_from_coord * p;
            v.into_homogeneous_point().normalized()
        });

        let world_from_view = Isometry3::new(
            scene.transform(camera.transform_ref).0.translation,
            self.view_drag.rotation,
        );
        let ray_origin = world_from_view.translation;
        let ray_vec_from_coord = world_from_view.rotation.into_matrix() * xy_from_coord.into_homogeneous_matrix();

        // start render
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(ui.key_index(Key::Escape)) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                let ui = &ui;
                let accel = &self.accel;
                let selected_camera_ref = &mut self.camera_ref;
                let view_drag = &mut self.view_drag;
                let light = self.light.as_ref();
                move || {
                    for camera_ref in accel.scene().camera_ref_iter() {
                        if ui.radio_button(&im_str!("Camera {}", camera_ref.0), selected_camera_ref, camera_ref) {
                            let rotation = accel.scene().transform(camera.transform_ref).0.rotation;
                            *selected_camera_ref = camera_ref;
                            *view_drag = ViewDrag::new(rotation);
                        }
                    }
                    ui.text(format!("Lights: {}", light.map_or(0, |_| 1)));
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        self.accel.update(
            &base.context,
            &mut base.systems.resource_loader,
            &mut base.systems.global_allocator,
            &mut schedule,
        );

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

        let trace_image = self.accel.trace(
            &base.context,
            &base.systems.resource_loader,
            &mut schedule,
            &base.systems.descriptor_pool,
            &swap_extent,
            ray_origin,
            ray_vec_from_coord,
            self.light.as_ref(),
        );

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new(swap_image, &[0f32, 0f32, 0f32, 0f32]);

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
                            "trace/copy.vert.spv",
                            "trace/copy.frag.spv",
                            copy_descriptor_set,
                            3,
                        );
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
    }
}

fn main() {
    let mut params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        allow_ray_tracing: true,
        ..Default::default()
    };
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        if !params.parse_arg(arg) {
            panic!("unknown argument {:?}", arg);
        }
    }

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("trace")
        .with_inner_size(Size::Logical(LogicalSize::new(640.0, 640.0)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &params);
    let app = App::new(&mut base);

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
