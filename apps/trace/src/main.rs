mod accel;
mod blender;
mod renderer;
mod scene;
mod tungsten;

use crate::renderer::*;
use crate::scene::*;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::*;
use imgui::{im_str, CollapsingHeader, Drag, Key, MouseButton};
use spark::vk;
use std::{env, ops::Deref, sync::Arc};
use winit::{
    dpi::{LogicalSize, Size},
    event::VirtualKeyCode,
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CopyData {
    exposure_scale: f32,
    render_color_space: u32,
    tone_map_method: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Contiguous, Eq, PartialEq)]
enum ToneMapMethod {
    None = 0,
    FilmicSrgb = 1,
    AcesFit = 2,
}

descriptor_set_layout!(CopyDescriptorSetLayout {
    data: UniformData<CopyData>,
    result: StorageImage,
});

struct ViewAdjust {
    translation: Vec3,
    rotation: Rotor3,
    log2_scale: f32,
    drag_start: Option<(Vec2, Rotor3, Vec3)>,
}

impl ViewAdjust {
    fn new(world_from_camera: Similarity3) -> Self {
        Self {
            translation: world_from_camera.translation,
            rotation: world_from_camera.rotation,
            log2_scale: world_from_camera.scale.abs().log2(),
            drag_start: None,
        }
    }

    fn update(&mut self, io: &imgui::Io, fov_y: f32) -> bool {
        let mut was_updated = false;
        if io.want_capture_mouse {
            self.drag_start = None;
        } else {
            let display_size: Vec2 = io.display_size.into();
            let aspect_ratio = (display_size.x as f32) / (display_size.y as f32);

            let xy_from_st = Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * fov_y).tan(), Vec2::zero());
            let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, -2.0), Vec2::new(1.0, 1.0));
            let coord_from_uv = Scale2Offset2::new(display_size, Vec2::zero());
            let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

            let mouse_now = Vec2::from(io.mouse_pos);
            let dir_now = (xy_from_coord * mouse_now).into_homogeneous_point().normalized();
            if io[MouseButton::Left] {
                if let Some((mouse_start, rotation_start, dir_start)) = self.drag_start {
                    if (mouse_now - mouse_start).mag() > io.mouse_drag_threshold {
                        self.rotation = rotation_start * Rotor3::from_rotation_between(dir_now, dir_start);
                        was_updated = true;
                    }
                } else {
                    self.drag_start = Some((mouse_now, self.rotation, dir_now));
                }
            } else {
                if let Some((mouse_start, rotation_start, dir_start)) = self.drag_start.take() {
                    if (mouse_now - mouse_start).mag() > io.mouse_drag_threshold {
                        self.rotation = rotation_start * Rotor3::from_rotation_between(dir_now, dir_start);
                        was_updated = true;
                    }
                }
            }
        }
        if !io.want_capture_keyboard {
            let step_size = 5.0 * io.delta_time * self.log2_scale.exp();
            if io.keys_down[VirtualKeyCode::W as usize] {
                let v = if io.key_shift { Vec3::unit_y() } else { Vec3::unit_z() };
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if io.keys_down[VirtualKeyCode::S as usize] {
                let v = if io.key_shift { -Vec3::unit_y() } else { -Vec3::unit_z() };
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if io.keys_down[VirtualKeyCode::A as usize] {
                let v = Vec3::unit_x();
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if io.keys_down[VirtualKeyCode::D as usize] {
                let v = -Vec3::unit_x();
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
        }
        was_updated
    }

    fn world_from_camera(&self) -> Similarity3 {
        Similarity3::new(self.translation, self.rotation, self.log2_scale.exp2())
    }
}

struct App {
    context: Arc<Context>,

    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    scene: Arc<Scene>,
    renderer: Renderer,
    renderer_state: RendererState,

    show_debug_ui: bool,
    log2_exposure_scale: f32,
    tone_map_method: ToneMapMethod,
    view_adjust: ViewAdjust,
    fov_y: f32,
}

impl App {
    fn new(base: &mut AppBase, scene: Scene, renderer_params: RendererParams) -> Self {
        let context = &base.context;
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(descriptor_set_layout_cache);
        let copy_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(copy_descriptor_set_layout.0);

        let scene = Arc::new(scene);
        let renderer = Renderer::new(
            &context,
            &scene,
            &mut base.systems.descriptor_set_layout_cache,
            &base.systems.pipeline_cache,
            &mut base.systems.resource_loader,
            &mut base.systems.render_graph,
            &mut base.systems.global_allocator,
            renderer_params,
        );
        let renderer_state = RendererState::new();

        let camera = scene.cameras.first().unwrap();
        let world_from_camera = scene.transform(camera.transform_ref).world_from_local;
        let fov_y = camera.fov_y;

        Self {
            context: Arc::clone(&context),
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            scene,
            renderer,
            renderer_state,
            show_debug_ui: true,
            log2_exposure_scale: 0f32,
            tone_map_method: ToneMapMethod::AcesFit,
            view_adjust: ViewAdjust::new(world_from_camera),
            fov_y,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        // TODO: move to an update function
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(ui.key_index(Key::Escape)) {
            base.exit_requested = true;
        }
        if ui.is_mouse_clicked(MouseButton::Right) {
            self.show_debug_ui = !self.show_debug_ui;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                || {
                    if CollapsingHeader::new(im_str!("Renderer")).default_open(true).build(&ui) {
                        self.renderer.debug_ui(&mut self.renderer_state, &ui);
                    }
                    if CollapsingHeader::new(im_str!("Film")).default_open(true).build(&ui) {
                        let id = ui.push_id(im_str!("Tone Map"));
                        Drag::new(im_str!("Exposure Bias"))
                            .speed(0.05)
                            .build(&ui, &mut self.log2_exposure_scale);
                        ui.text("Tone Map:");
                        ui.radio_button(im_str!("None"), &mut self.tone_map_method, ToneMapMethod::None);
                        ui.radio_button(
                            im_str!("Filmic sRGB"),
                            &mut self.tone_map_method,
                            ToneMapMethod::FilmicSrgb,
                        );
                        ui.radio_button(
                            im_str!("ACES (fitted)"),
                            &mut self.tone_map_method,
                            ToneMapMethod::AcesFit,
                        );
                        id.pop(&ui);
                    }
                    let mut needs_reset = false;
                    if CollapsingHeader::new(im_str!("Scene")).default_open(true).build(&ui) {
                        let scene = self.scene.deref();
                        ui.text("Cameras:");
                        for camera_ref in scene.camera_ref_iter() {
                            if ui.small_button(&im_str!("Camera {}", camera_ref.0)) {
                                let camera = scene.camera(camera_ref);
                                let world_from_camera = scene.transform(camera.transform_ref).world_from_local;
                                self.view_adjust = ViewAdjust::new(world_from_camera);
                                self.fov_y = camera.fov_y;
                                needs_reset = true;
                            }
                        }
                        needs_reset |= Drag::new(im_str!("Camera FOV"))
                            .speed(0.005)
                            .build(&ui, &mut self.fov_y);
                        Drag::new(im_str!("Camera Scale Bias"))
                            .speed(0.05)
                            .build(&ui, &mut self.view_adjust.log2_scale);
                        ui.text(format!(
                            "Lights: {}",
                            scene.lights.len()
                                + scene
                                    .instances
                                    .iter()
                                    .filter_map(|instance| scene.material(instance.material_ref).emission)
                                    .count()
                        ));
                    }
                    if needs_reset {
                        self.renderer_state.reset_image();
                    }
                }
            });

        if self.view_adjust.update(ui.io(), self.fov_y) {
            self.renderer_state.reset_image();
        }
        let world_from_camera = self.view_adjust.world_from_camera();

        // start render
        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        self.renderer.update(
            &base.context,
            &mut schedule,
            &mut base.systems.resource_loader,
            &mut base.systems.global_allocator,
        );
        let result_image = self.renderer.render(
            &mut self.renderer_state,
            &base.context,
            &mut schedule,
            &base.systems.pipeline_cache,
            &base.systems.descriptor_pool,
            &base.systems.resource_loader,
            world_from_camera,
            self.fov_y,
        );

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore);
        let swap_size = base.display.swapchain.get_size();
        let swap_format = base.display.swapchain.get_format();
        let swap_image = schedule.import_image(
            &ImageDesc::new_2d(swap_size, swap_format, vk::ImageAspectFlags::COLOR),
            ImageUsage::COLOR_ATTACHMENT_WRITE | ImageUsage::SWAPCHAIN,
            swap_vk_image,
            ImageUsage::empty(),
        );

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new(swap_image, &[0f32, 0f32, 0f32, 0f32]);

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if let Some(result_image) = result_image {
                    params.add_image(result_image, ImageUsage::FRAGMENT_STORAGE_READ);
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
                let show_debug_ui = self.show_debug_ui;
                let log2_exposure_scale = self.log2_exposure_scale;
                let render_color_space = self.renderer.params.render_color_space;
                let tone_map_method = self.tone_map_method;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);

                    if let Some(result_image) = result_image {
                        let result_image_view = params.get_image_view(result_image);

                        let copy_descriptor_set = copy_descriptor_set_layout.write(
                            &descriptor_pool,
                            |buf: &mut CopyData| {
                                *buf = CopyData {
                                    exposure_scale: log2_exposure_scale.exp2(),
                                    render_color_space: render_color_space.into_integer(),
                                    tone_map_method: tone_map_method.into_integer(),
                                }
                            },
                            result_image_view,
                        );

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
                    if show_debug_ui {
                        let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                        ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
                    }
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

struct CommandlineApp {
    context: Arc<Context>,
    systems: AppSystems,
}

impl CommandlineApp {
    fn new(params: &ContextParams) -> Self {
        let context = Arc::new(Context::new(None, params));
        let systems = AppSystems::new(&context);
        Self { context, systems }
    }
}

enum SceneDesc {
    CornellBox(CornellBoxVariant),
    BlenderExport(String),
    Tungsten(String),
}

fn main() {
    let mut context_params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        ray_tracing: ContextFeature::Required,
        ..Default::default()
    };
    let mut renderer_params = RendererParams::default();
    let mut scene_desc = SceneDesc::CornellBox(CornellBoxVariant::Original);
    let mut run_commandline_app = false;
    {
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-c" => run_commandline_app = true,
                "-b" => renderer_params.max_bounces = it.next().and_then(|s| s.as_str().parse::<u32>().ok()).unwrap(),
                "-s" => {
                    scene_desc = match it.next().unwrap().as_str() {
                        "cornell" => SceneDesc::CornellBox(CornellBoxVariant::Original),
                        "cornell-mirror" => SceneDesc::CornellBox(CornellBoxVariant::Mirror),
                        "cornell-conductor" => SceneDesc::CornellBox(CornellBoxVariant::Conductor),
                        "cornell-instances" => SceneDesc::CornellBox(CornellBoxVariant::Instances),
                        "cornell-domelight" => SceneDesc::CornellBox(CornellBoxVariant::DomeLight),
                        "cornell-spherelight" => SceneDesc::CornellBox(CornellBoxVariant::SphereLight),
                        s => panic!("unknown scene {:?}", s),
                    }
                }
                "-f" => {
                    scene_desc = SceneDesc::BlenderExport(it.next().unwrap());
                }
                "-t" => {
                    scene_desc = SceneDesc::Tungsten(it.next().unwrap());
                }
                _ => {
                    if !context_params.parse_arg(arg.as_str()) {
                        panic!("unknown argument {:?}", arg);
                    }
                }
            }
        }
    }

    let scene = match scene_desc {
        SceneDesc::CornellBox(variant) => create_cornell_box_scene(&variant),
        SceneDesc::BlenderExport(filename) => {
            let contents = std::fs::read_to_string(filename.as_str()).unwrap();
            blender::load_export(&contents)
        }
        SceneDesc::Tungsten(filename) => tungsten::load_scene(filename),
    };

    if run_commandline_app {
        let app = CommandlineApp::new(&context_params);

        // TODO
    } else {
        let event_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("trace")
            .with_inner_size(Size::Logical(LogicalSize::new(
                renderer_params.size.x as f64,
                renderer_params.size.y as f64,
            )))
            .build(&event_loop)
            .unwrap();

        let mut base = AppBase::new(window, &context_params);
        let app = App::new(&mut base, scene, renderer_params);

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
}
