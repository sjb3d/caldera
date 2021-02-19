mod accel;
mod import;
mod renderer;
mod scene;
mod tungsten;

use crate::renderer::*;
use crate::scene::*;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::*;
use imgui::{im_str, CollapsingHeader, Drag, Key, MouseButton};
use spark::vk;
use std::{
    ffi::CString,
    ops::Deref,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};
use structopt::StructOpt;
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

descriptor_set_layout!(CopyDescriptorSetLayout {
    data: UniformData<CopyData>,
    result: StorageImage,
});

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CaptureData {
    size: UVec2,
    exposure_scale: f32,
    render_color_space: u32,
    tone_map_method: u32,
}

descriptor_set_layout!(CaptureDescriptorSetLayout {
    data: UniformData<CaptureData>,
    output: StorageBuffer,
    input: StorageImage,
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
    progress: RenderProgress,

    show_debug_ui: bool,
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
        let progress = RenderProgress::new();

        let camera = scene.cameras.first().unwrap();
        let world_from_camera = scene.transform(camera.transform_ref).world_from_local;
        let fov_y = renderer.params.fov_y_override.unwrap_or(camera.fov_y);

        Self {
            context: Arc::clone(&context),
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            scene,
            renderer,
            progress,
            show_debug_ui: true,
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
                    self.renderer.debug_ui(&mut self.progress, &ui);
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
                        self.progress.reset();
                    }
                }
            });

        if self.view_adjust.update(ui.io(), self.fov_y) {
            self.progress.reset();
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
            &mut self.progress,
            &base.context,
            &mut schedule,
            &base.systems.pipeline_cache,
            &base.systems.descriptor_pool,
            &base.systems.resource_loader,
            world_from_camera,
            self.fov_y,
        );

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore.unwrap());
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
                let renderer_params = &self.renderer.params;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);

                    if let Some(result_image) = result_image {
                        let result_image_view = params.get_image_view(result_image);

                        let copy_descriptor_set = copy_descriptor_set_layout.write(
                            &descriptor_pool,
                            |buf: &mut CopyData| {
                                *buf = CopyData {
                                    exposure_scale: renderer_params.log2_exposure_scale.exp2(),
                                    render_color_space: renderer_params.render_color_space.into_integer(),
                                    tone_map_method: renderer_params.tone_map_method.into_integer(),
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
            Some(swap_image),
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display
            .present(swap_vk_image, rendering_finished_semaphore.unwrap());
    }
}

struct CaptureBuffer {
    context: Arc<Context>,
    size: u32,
    mem: vk::DeviceMemory,
    buffer: UniqueBuffer,
    mapping: *const u8,
}

impl CaptureBuffer {
    fn new(context: &Arc<Context>, size: u32) -> Self {
        let buffer = {
            let create_info = vk::BufferCreateInfo {
                size: size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                ..Default::default()
            };
            unsafe { context.device.create_buffer(&create_info, None) }.unwrap()
        };

        let mem_req = unsafe { context.device.get_buffer_memory_requirements(buffer) };

        let mem = {
            let memory_type_index = context
                .get_memory_type_index(mem_req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE)
                .unwrap();
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: size as vk::DeviceSize,
                memory_type_index,
                ..Default::default()
            };
            unsafe { context.device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };

        unsafe { context.device.bind_buffer_memory(buffer, mem, 0) }.unwrap();

        let mapping = unsafe { context.device.map_memory(mem, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();

        Self {
            context: Arc::clone(&context),
            size,
            mem,
            buffer: Unique::new(buffer, context.allocate_handle_uid()),
            mapping: mapping as *const _,
        }
    }

    fn mapping(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.mapping, self.size as usize) }
    }
}

impl Drop for CaptureBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(Some(self.buffer.0), None);
            self.context.device.unmap_memory(self.mem);
            self.context.device.free_memory(Some(self.mem), None);
        }
    }
}

struct CommandlineApp {
    context: Arc<Context>,
    systems: AppSystems,
    scene: Arc<Scene>,
    renderer: Renderer,
    progress: RenderProgress,

    capture_descriptor_set_layout: CaptureDescriptorSetLayout,
    capture_pipeline_layout: vk::PipelineLayout,
    capture_buffer: CaptureBuffer,
}

impl CommandlineApp {
    fn new(context_params: &ContextParams, scene: Scene, renderer_params: RendererParams) -> Self {
        let context = Arc::new(Context::new(None, context_params));
        let mut systems = AppSystems::new(&context);

        let scene = Arc::new(scene);

        let renderer = Renderer::new(
            &context,
            &scene,
            &mut systems.descriptor_set_layout_cache,
            &systems.pipeline_cache,
            &mut systems.resource_loader,
            &mut systems.render_graph,
            &mut systems.global_allocator,
            renderer_params,
        );

        let progress = RenderProgress::new();

        let capture_descriptor_set_layout = CaptureDescriptorSetLayout::new(&mut systems.descriptor_set_layout_cache);
        let capture_pipeline_layout = systems
            .descriptor_set_layout_cache
            .create_pipeline_layout(capture_descriptor_set_layout.0);

        let capture_buffer = CaptureBuffer::new(&context, renderer.params.width * renderer.params.height * 3);

        Self {
            context,
            systems,
            scene,
            renderer,
            progress,
            capture_descriptor_set_layout,
            capture_pipeline_layout,
            capture_buffer,
        }
    }

    fn run(&mut self, filename: &Path) {
        let camera = self.scene.cameras.first().unwrap();
        let world_from_camera = self.scene.transform(camera.transform_ref).world_from_local;
        let fov_y = self.renderer.params.fov_y_override.unwrap_or(camera.fov_y);

        let mut render_started = false;
        loop {
            let cbar = self.systems.acquire_command_buffer();

            let mut schedule = RenderSchedule::new(&mut self.systems.render_graph);

            self.renderer.update(
                &self.context,
                &mut schedule,
                &mut self.systems.resource_loader,
                &mut self.systems.global_allocator,
            );

            let mut copy_done = false;
            if let Some(result_image) = self.renderer.render(
                &mut self.progress,
                &self.context,
                &mut schedule,
                &self.systems.pipeline_cache,
                &self.systems.descriptor_pool,
                &self.systems.resource_loader,
                world_from_camera,
                fov_y,
            ) {
                if !render_started {
                    println!("starting render");
                    render_started = true;
                }

                if self.progress.done(&self.renderer.params) {
                    let capture_desc = BufferDesc::new(self.capture_buffer.size as usize);
                    let capture_buffer = schedule.import_buffer(
                        &capture_desc,
                        BufferUsage::COMPUTE_STORAGE_WRITE,
                        self.capture_buffer.buffer,
                        BufferUsage::empty(),
                    );

                    schedule.add_compute(
                        command_name!("capture"),
                        |params| {
                            params.add_image(result_image, ImageUsage::COMPUTE_STORAGE_READ);
                            params.add_buffer(capture_buffer, BufferUsage::COMPUTE_STORAGE_WRITE);
                        },
                        {
                            let capture_descriptor_set_layout = &self.capture_descriptor_set_layout;
                            let capture_pipeline_layout = self.capture_pipeline_layout;
                            let pipeline_cache = &self.systems.pipeline_cache;
                            let descriptor_pool = &self.systems.descriptor_pool;
                            let context = &self.context;
                            let renderer_params = &self.renderer.params;
                            move |params, cmd| {
                                let result_image_view = params.get_image_view(result_image);
                                let capture_buffer = params.get_buffer(capture_buffer);

                                let descriptor_set = capture_descriptor_set_layout.write(
                                    &descriptor_pool,
                                    |buf: &mut CaptureData| {
                                        *buf = CaptureData {
                                            size: renderer_params.size(),
                                            exposure_scale: renderer_params.log2_exposure_scale.exp2(),
                                            render_color_space: renderer_params.render_color_space.into_integer(),
                                            tone_map_method: renderer_params.tone_map_method.into_integer(),
                                        };
                                    },
                                    capture_buffer,
                                    result_image_view,
                                );

                                dispatch_helper(
                                    &context.device,
                                    pipeline_cache,
                                    cmd,
                                    capture_pipeline_layout,
                                    "trace/capture.comp.spv",
                                    descriptor_set,
                                    renderer_params.size().div_round_up(16),
                                );
                            }
                        },
                    );

                    copy_done = true;
                }
            } else {
                // waiting for async load
                std::thread::sleep(std::time::Duration::from_millis(5));
            }

            schedule.run(
                &self.context,
                cbar.pre_swapchain_cmd,
                cbar.post_swapchain_cmd,
                None,
                &mut self.systems.query_pool,
            );

            self.systems.submit_command_buffer(&cbar);

            if copy_done {
                break;
            }
        }

        println!("waiting for idle");
        unsafe { self.context.device.device_wait_idle() }.unwrap();

        println!("saving image to {:?}", filename);
        match filename.extension().unwrap().to_str().unwrap() {
            "png" => {
                stb::image_write::stbi_write_png(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    self.renderer.params.width as i32,
                    self.renderer.params.height as i32,
                    3,
                    self.capture_buffer.mapping(),
                    (self.renderer.params.width * 3) as i32,
                )
                .unwrap();
            }
            "tga" => {
                stb::image_write::stbi_write_tga(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    self.renderer.params.width as i32,
                    self.renderer.params.height as i32,
                    3,
                    self.capture_buffer.mapping(),
                )
                .unwrap();
            }
            "jpg" => {
                let quality = 95;
                stb::image_write::stbi_write_jpg(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    self.renderer.params.width as i32,
                    self.renderer.params.height as i32,
                    3,
                    self.capture_buffer.mapping(),
                    quality,
                )
                .unwrap();
            }
            _ => panic!("unknown extension"),
        }

        println!("shutting down");
    }
}

#[derive(Debug, StructOpt)]
enum SceneDesc {
    /// Load a cornell box scene
    CornellBox {
        #[structopt(possible_values=&CornellBoxVariant::VARIANTS, default_value="original")]
        variant: CornellBoxVariant,
    },
    /// Import from .caldera file
    Import { filename: PathBuf },
    /// Import from Tungsten scene.json file
    Tungsten { filename: PathBuf },
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    /// Core Vulkan version to load
    #[structopt(short, long, parse(try_from_str=try_version_from_str), default_value="1.1", global=true)]
    version: vk::Version,

    /// Whether to use EXT_inline_uniform_block
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional", global=true)]
    inline_uniform_block: ContextFeature,

    /// Run without a window and output to file
    #[structopt(short, long, global = true, display_order = 3)]
    output: Option<PathBuf>,

    #[structopt(flatten)]
    renderer_params: RendererParams,

    #[structopt(subcommand)]
    scene_desc: Option<SceneDesc>,
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        ray_tracing: ContextFeature::Require,
        ..Default::default()
    };
    let renderer_params = app_params.renderer_params;

    let scene = match app_params.scene_desc.as_ref().unwrap_or(&SceneDesc::CornellBox {
        variant: CornellBoxVariant::Original,
    }) {
        SceneDesc::CornellBox { variant } => create_cornell_box_scene(variant),
        SceneDesc::Import { filename } => {
            let contents = std::fs::read_to_string(filename).unwrap();
            import::load_scene(&contents)
        }
        SceneDesc::Tungsten { filename } => tungsten::load_scene(filename),
    };

    if let Some(output) = app_params.output {
        let mut app = CommandlineApp::new(&context_params, scene, renderer_params);
        app.run(&output);
    } else {
        let event_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("trace")
            .with_inner_size(Size::Logical(LogicalSize::new(
                renderer_params.width as f64,
                renderer_params.height as f64,
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
