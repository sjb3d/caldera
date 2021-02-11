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
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use spark::vk;
use std::sync::Arc;
use std::{env, ops::Deref};
use winit::{
    dpi::{LogicalSize, Size},
    event::VirtualKeyCode,
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SamplePixel {
    x: u16,
    y: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CopyData {
    sample_scale: f32,
    render_color_space: u32,
    tone_map_method: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Contiguous, Eq, PartialEq)]
pub enum RenderColorSpace {
    Rec709 = 0,
    ACEScg = 1,
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
    result_r: StorageImage,
    result_g: StorageImage,
    result_b: StorageImage,
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

    sample_image: StaticImageHandle,
    result_images: (ImageHandle, ImageHandle, ImageHandle),
    next_sample_index: u32,

    scene: Arc<Scene>,
    renderer: Renderer,
    camera_ref: CameraRef,

    log2_exposure_scale: f32,
    render_color_space: RenderColorSpace,
    tone_map_method: ToneMapMethod,
    view_adjust: ViewAdjust,
}

impl App {
    const SEQUENCE_COUNT: u32 = 1024;
    const SAMPLES_PER_SEQUENCE: u32 = 256;

    fn trace_size() -> UVec2 {
        UVec2::new(1920, 1080)
    }

    fn new(base: &mut AppBase, params: &AppParams) -> Self {
        let context = &base.context;
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(descriptor_set_layout_cache);
        let copy_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(copy_descriptor_set_layout.0);

        let sample_image = base.systems.resource_loader.create_image();
        base.systems.resource_loader.async_load(move |allocator| {
            let sequences: Vec<Vec<_>> = (0..Self::SEQUENCE_COUNT)
                .into_par_iter()
                .map(|i| {
                    let mut rng = SmallRng::seed_from_u64(i as u64);
                    pmj::generate(Self::SAMPLES_PER_SEQUENCE as usize, 4, &mut rng)
                })
                .collect();

            let desc = ImageDesc::new_2d(
                UVec2::new(Self::SAMPLES_PER_SEQUENCE, Self::SEQUENCE_COUNT),
                vk::Format::R16G16_UINT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = allocator
                .map_image(sample_image, &desc, ImageUsage::COMPUTE_STORAGE_READ)
                .unwrap();

            for sample in sequences.iter().flat_map(|sequence| sequence.iter()) {
                let pixel = SamplePixel {
                    x: sample.x_bits(16) as u16,
                    y: sample.y_bits(16) as u16,
                };
                writer.write(&pixel);
            }
        });

        let result_images = {
            let trace_size = Self::trace_size();
            let desc = ImageDesc::new_2d(trace_size, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR);
            let usage = ImageUsage::FRAGMENT_STORAGE_READ
                | ImageUsage::RAY_TRACING_STORAGE_READ
                | ImageUsage::RAY_TRACING_STORAGE_WRITE;
            let render_graph = &mut base.systems.render_graph;
            let global_allocator = &mut base.systems.global_allocator;
            (
                render_graph.create_image(&desc, usage, global_allocator),
                render_graph.create_image(&desc, usage, global_allocator),
                render_graph.create_image(&desc, usage, global_allocator),
            )
        };

        let scene = Arc::new(match &params.scene_desc {
            SceneDesc::CornellBox(variant) => create_cornell_box_scene(variant),
            SceneDesc::BlenderExport(filename) => {
                let contents = std::fs::read_to_string(filename.as_str()).unwrap();
                blender::load_export(&contents)
            }
            SceneDesc::Tungsten(filename) => tungsten::load_scene(filename),
        });
        let renderer = Renderer::new(
            &context,
            &scene,
            &mut base.systems.descriptor_set_layout_cache,
            &base.systems.pipeline_cache,
            &mut base.systems.resource_loader,
            params.max_bounces,
        );

        let camera_ref = CameraRef(0);
        let world_from_camera = scene.transform(scene.camera(camera_ref).transform_ref).world_from_local;

        Self {
            context: Arc::clone(&context),
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            sample_image,
            result_images,
            next_sample_index: 0,
            scene,
            renderer,
            camera_ref,
            log2_exposure_scale: 0f32,
            render_color_space: RenderColorSpace::ACEScg,
            tone_map_method: ToneMapMethod::AcesFit,
            view_adjust: ViewAdjust::new(world_from_camera),
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        // TODO: move to an update function
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(ui.key_index(Key::Escape)) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                || {
                    let mut needs_reset = false;
                    if CollapsingHeader::new(im_str!("Renderer")).default_open(true).build(&ui) {
                        ui.text("Color Space:");
                        needs_reset |= ui.radio_button(
                            im_str!("Rec709 (sRGB primaries)"),
                            &mut self.render_color_space,
                            RenderColorSpace::Rec709,
                        );
                        needs_reset |= ui.radio_button(
                            im_str!("ACEScg (AP1 primaries)"),
                            &mut self.render_color_space,
                            RenderColorSpace::ACEScg,
                        );
                        needs_reset |= self.renderer.debug_ui(&ui);
                    }
                    if CollapsingHeader::new(im_str!("Tone Map")).default_open(true).build(&ui) {
                        let id = ui.push_id(im_str!("Tone Map"));
                        Drag::new(im_str!("Exposure Bias"))
                            .speed(0.05)
                            .build(&ui, &mut self.log2_exposure_scale);
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
                    if CollapsingHeader::new(im_str!("Scene")).default_open(true).build(&ui) {
                        let scene = self.scene.deref();
                        ui.text("Cameras:");
                        for camera_ref in scene.camera_ref_iter() {
                            if ui.radio_button(&im_str!("Camera {}", camera_ref.0), &mut self.camera_ref, camera_ref) {
                                let camera = scene.camera(camera_ref);
                                let world_from_camera = scene.transform(camera.transform_ref).world_from_local;
                                self.camera_ref = camera_ref;
                                self.view_adjust = ViewAdjust::new(world_from_camera);
                                needs_reset = true;
                            }
                        }
                        needs_reset |= Drag::new(im_str!("Camera Scale Bias"))
                            .speed(0.05)
                            .build(&ui, &mut self.view_adjust.log2_scale);
                        ui.text(format!(
                            "Lights: {}",
                            scene.lights.len()
                                + scene
                                    .instances
                                    .iter()
                                    .filter_map(|instance| scene.shader(instance.shader_ref).emission)
                                    .count()
                        ));
                    }

                    if needs_reset {
                        self.next_sample_index = 0;
                    }
                }
            });

        let camera = self.scene.camera(self.camera_ref);
        let fov_y = camera.fov_y;
        if self.view_adjust.update(ui.io(), fov_y) {
            self.next_sample_index = 0;
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
            &mut base.systems.resource_loader,
            &mut base.systems.global_allocator,
            &mut schedule,
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

        let next_sample_index = if self.next_sample_index == Self::SAMPLES_PER_SEQUENCE {
            self.next_sample_index
        } else if let (Some(sample_image_view), true) = (
            base.systems.resource_loader.get_image_view(self.sample_image),
            self.renderer.is_ready(&base.systems.resource_loader),
        ) {
            let trace_size = Self::trace_size();

            schedule.add_compute(
                command_name!("trace"),
                |params| {
                    self.renderer.declare_parameters(params);
                    params.add_image(
                        self.result_images.0,
                        ImageUsage::RAY_TRACING_STORAGE_READ | ImageUsage::RAY_TRACING_STORAGE_WRITE,
                    );
                    params.add_image(
                        self.result_images.1,
                        ImageUsage::RAY_TRACING_STORAGE_READ | ImageUsage::RAY_TRACING_STORAGE_WRITE,
                    );
                    params.add_image(
                        self.result_images.2,
                        ImageUsage::RAY_TRACING_STORAGE_READ | ImageUsage::RAY_TRACING_STORAGE_WRITE,
                    );
                },
                {
                    let descriptor_pool = &base.systems.descriptor_pool;
                    let resource_loader = &base.systems.resource_loader;
                    let render_color_space = self.render_color_space;
                    let renderer = &self.renderer;
                    let next_sample_index = self.next_sample_index;
                    let camera_ref = self.camera_ref;
                    let result_images = &self.result_images;
                    move |params, cmd| {
                        let result_image_views = (
                            params.get_image_view(result_images.0),
                            params.get_image_view(result_images.1),
                            params.get_image_view(result_images.2),
                        );

                        renderer.emit_trace(
                            cmd,
                            descriptor_pool,
                            resource_loader,
                            render_color_space,
                            sample_image_view,
                            next_sample_index,
                            camera_ref,
                            world_from_camera,
                            &result_image_views,
                            trace_size,
                        );
                    }
                },
            );

            self.next_sample_index + 1
        } else {
            0
        };

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new(swap_image, &[0f32, 0f32, 0f32, 0f32]);

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if next_sample_index != 0 {
                    params.add_image(self.result_images.0, ImageUsage::FRAGMENT_STORAGE_READ);
                    params.add_image(self.result_images.1, ImageUsage::FRAGMENT_STORAGE_READ);
                    params.add_image(self.result_images.2, ImageUsage::FRAGMENT_STORAGE_READ);
                }
            },
            {
                let context = &base.context;
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let copy_descriptor_set_layout = &self.copy_descriptor_set_layout;
                let copy_pipeline_layout = self.copy_pipeline_layout;
                let result_images = &self.result_images;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                let log2_exposure_scale = self.log2_exposure_scale;
                let render_color_space = self.render_color_space;
                let tone_map_method = self.tone_map_method;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);

                    if next_sample_index != 0 {
                        let result_image_views = (
                            params.get_image_view(result_images.0),
                            params.get_image_view(result_images.1),
                            params.get_image_view(result_images.2),
                        );

                        let copy_descriptor_set = copy_descriptor_set_layout.write(
                            &descriptor_pool,
                            |buf: &mut CopyData| {
                                *buf = CopyData {
                                    sample_scale: log2_exposure_scale.exp2() / (next_sample_index as f32),
                                    render_color_space: render_color_space.into_integer(),
                                    tone_map_method: tone_map_method.into_integer(),
                                }
                            },
                            result_image_views.0,
                            result_image_views.1,
                            result_image_views.2,
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

        self.next_sample_index = next_sample_index;
    }
}

enum SceneDesc {
    CornellBox(CornellBoxVariant),
    BlenderExport(String),
    Tungsten(String),
}

struct AppParams {
    max_bounces: u32,
    scene_desc: SceneDesc,
}

impl Default for AppParams {
    fn default() -> Self {
        Self {
            max_bounces: 2,
            scene_desc: SceneDesc::CornellBox(CornellBoxVariant::Original),
        }
    }
}

fn main() {
    let mut context_params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        ray_tracing: ContextFeature::Required,
        ..Default::default()
    };
    let mut app_params = AppParams::default();
    {
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-b" => app_params.max_bounces = it.next().and_then(|s| s.as_str().parse::<u32>().ok()).unwrap(),
                "-s" => {
                    app_params.scene_desc = match it.next().unwrap().as_str() {
                        "cornell" => SceneDesc::CornellBox(CornellBoxVariant::Original),
                        "cornell-mirror" => SceneDesc::CornellBox(CornellBoxVariant::Mirror),
                        "cornell-ggx" => SceneDesc::CornellBox(CornellBoxVariant::GGX),
                        "cornell-instances" => SceneDesc::CornellBox(CornellBoxVariant::Instances),
                        "cornell-domelight" => SceneDesc::CornellBox(CornellBoxVariant::DomeLight),
                        "cornell-spherelight" => SceneDesc::CornellBox(CornellBoxVariant::SphereLight),
                        s => panic!("unknown scene {:?}", s),
                    }
                }
                "-f" => {
                    app_params.scene_desc = SceneDesc::BlenderExport(it.next().unwrap());
                }
                "-t" => {
                    app_params.scene_desc = SceneDesc::Tungsten(it.next().unwrap());
                }
                _ => {
                    if !context_params.parse_arg(arg.as_str()) {
                        panic!("unknown argument {:?}", arg);
                    }
                }
            }
        }
    }

    let event_loop = EventLoop::new();

    let size = App::trace_size();
    let window = WindowBuilder::new()
        .with_title("trace")
        .with_inner_size(Size::Logical(LogicalSize::new(size.x as f64, size.y as f64)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &context_params);
    let app = App::new(&mut base, &app_params);

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
