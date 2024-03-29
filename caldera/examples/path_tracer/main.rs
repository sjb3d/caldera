mod accel;
mod import;
mod renderer;
mod scene;
mod sobol;
mod spectrum;
mod tungsten;

mod prelude {
    pub use crate::sobol::*;
    pub use crate::spectrum::*;
}

use crate::{renderer::*, scene::*};
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::prelude::*;
use spark::vk;
use std::{
    ffi::CString,
    ops::Deref,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
    time::Instant,
};
use structopt::StructOpt;
use strum::{EnumString, EnumVariantNames, VariantNames};
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::EventLoop,
    monitor::VideoMode,
    window::{Fullscreen, WindowBuilder},
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CopyData {
    exposure_scale: f32,
    rec709_from_xyz: Mat3,
    acescg_from_xyz: Mat3,
    tone_map_method: u32,
}

descriptor_set!(CopyDescriptorSet {
    data: UniformData<CopyData>,
    result: StorageImage,
});

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CaptureData {
    size: UVec2,
    exposure_scale: f32,
    rec709_from_xyz: Mat3,
    acescg_from_xyz: Mat3,
    tone_map_method: u32,
}

descriptor_set!(CaptureDescriptorSet {
    data: UniformData<CaptureData>,
    output: StorageBuffer,
    input: StorageImage,
});

struct ViewAdjust {
    translation: Vec3,
    rotation: Rotor3,
    log2_scale: f32,
    drag_start: Option<(Rotor3, Vec3)>,
    fov_y: f32,
    aperture_radius: f32,
    focus_distance: f32,
}

impl ViewAdjust {
    fn new(camera: &Camera, fov_y_override: Option<f32>) -> Self {
        let (world_from_camera, fov_y, aperture_radius, focus_distance) = match *camera {
            Camera::Pinhole {
                world_from_camera,
                fov_y,
            } => (world_from_camera, fov_y, 0.0, 2.0),
            Camera::ThinLens {
                world_from_camera,
                fov_y,
                aperture_radius,
                focus_distance,
            } => (world_from_camera, fov_y, aperture_radius, focus_distance),
        };

        Self {
            translation: world_from_camera.translation,
            rotation: world_from_camera.rotation,
            log2_scale: world_from_camera.scale.abs().log2(),
            drag_start: None,
            fov_y: fov_y_override.unwrap_or(fov_y),
            aperture_radius,
            focus_distance,
        }
    }

    fn update(&mut self, response: egui::Response) -> bool {
        let mut was_updated = false;
        {
            let origin = Vec2::new(response.rect.min.x, response.rect.min.y);
            let size = Vec2::new(response.rect.width(), response.rect.height());
            let aspect_ratio = (size.x as f32) / (size.y as f32);

            let xy_from_st = Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * self.fov_y).tan(), Vec2::zero());
            let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, -2.0), Vec2::new(1.0, 1.0));
            let coord_from_uv = Scale2Offset2::new(size, origin);
            let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

            let dir_from_coord = |coord: egui::Pos2| {
                let coord = Vec2::new(coord.x, coord.y);
                (xy_from_coord * coord).into_homogeneous_point().normalized()
            };

            if response.drag_started_by(egui::PointerButton::Primary) {
                self.drag_start = response
                    .interact_pointer_pos()
                    .map(|coord| (self.rotation, dir_from_coord(coord)));
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some((rotation_start, dir_start)) = self.drag_start {
                    if let Some(coord_now) = response.ctx.input(|i| i.pointer.latest_pos()) {
                        let dir_now = dir_from_coord(coord_now);
                        self.rotation = rotation_start * Rotor3::from_rotation_between(dir_now, dir_start);
                        was_updated = true;
                    }
                }
            }
        }
        response.ctx.input(|i| {
            let step_size = 5.0 * i.stable_dt * self.log2_scale.exp();
            if i.key_down(egui::Key::W) {
                let v = if i.modifiers.shift {
                    Vec3::unit_y()
                } else {
                    Vec3::unit_z()
                };
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if i.key_down(egui::Key::S) {
                let v = if i.modifiers.shift {
                    -Vec3::unit_y()
                } else {
                    -Vec3::unit_z()
                };
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if i.key_down(egui::Key::A) {
                let v = Vec3::unit_x();
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
            if i.key_down(egui::Key::D) {
                let v = -Vec3::unit_x();
                self.translation += step_size * (self.rotation * v);
                was_updated = true;
            }
        });
        was_updated
    }

    fn to_camera(&self) -> Camera {
        Camera::ThinLens {
            world_from_camera: Similarity3::new(self.translation, self.rotation, 1.0),
            fov_y: self.fov_y,
            aperture_radius: self.aperture_radius,
            focus_distance: self.focus_distance,
        }
    }
}

struct App {
    scene: SharedScene,
    renderer: TaskOutput<Renderer>,
    progress: RenderProgress,

    show_debug_ui: bool,
    view_adjust: ViewAdjust,
}

impl App {
    fn new(base: &mut AppBase, scene: Scene, renderer_params: RendererParams) -> Self {
        let fov_y_override = renderer_params.fov_y_override;

        let scene = Arc::new(scene);
        let renderer = base.systems.task_system.spawn_task({
            Renderer::new(
                base.systems.resource_loader.clone(),
                Arc::clone(&scene),
                renderer_params,
            )
        });
        let progress = RenderProgress::new();

        let view_adjust = ViewAdjust::new(scene.cameras.first().unwrap(), fov_y_override);
        Self {
            scene,
            renderer,
            progress,
            show_debug_ui: true,
            view_adjust,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let cbar = base.systems.acquire_command_buffer();

        base.ui_begin_frame();
        base.egui_ctx.clone().input(|i| {
            if i.key_pressed(egui::Key::Escape) {
                base.exit_requested = true;
            }
            self.show_debug_ui ^= i.pointer.secondary_clicked();
        });

        egui::Window::new("Debug")
            .default_pos([5.0, 5.0])
            .default_size([350.0, 600.0])
            .vscroll(true)
            .show(&base.egui_ctx, |ui| {
                if let Some(renderer) = self.renderer.get_mut() {
                    renderer.debug_ui(&mut self.progress, ui);
                }
                let mut needs_reset = false;
                egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| {
                    let scene = self.scene.deref();
                    ui.label("Cameras:");
                    for camera_ref in scene.camera_ref_iter() {
                        if ui.small_button(format!("Camera {}", camera_ref.0)).clicked() {
                            self.view_adjust = ViewAdjust::new(
                                scene.camera(camera_ref),
                                if let Some(renderer) = self.renderer.get_mut() {
                                    renderer.params.fov_y_override
                                } else {
                                    None
                                },
                            );
                            needs_reset = true;
                        }
                    }
                    ui.add(
                        egui::DragValue::new(&mut self.view_adjust.log2_scale)
                            .speed(0.05)
                            .prefix("Camera Scale Bias: "),
                    );
                    needs_reset |= ui
                        .add(
                            egui::DragValue::new(&mut self.view_adjust.fov_y)
                                .speed(0.005)
                                .prefix("Camera FOV: "),
                        )
                        .changed();
                    needs_reset |= ui
                        .add(
                            egui::Slider::new(&mut self.view_adjust.aperture_radius, 0.0..=0.1)
                                .prefix("Aperture Radius: "),
                        )
                        .changed();
                    needs_reset |= ui
                        .add(
                            egui::Slider::new(&mut self.view_adjust.focus_distance, 0.0..=10.0)
                                .prefix("Focus Distance: "),
                        )
                        .changed();
                });
                if needs_reset {
                    self.progress.reset();
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(&base.egui_ctx, |ui| {
                let response = ui.allocate_response(ui.available_size(), egui::Sense::drag());
                if self.view_adjust.update(response) {
                    self.progress.reset();
                }
            });

        base.systems.draw_ui(&base.egui_ctx);
        base.ui_end_frame(cbar.pre_swapchain_cmd);

        // start render
        let mut schedule = base.systems.resource_loader.begin_schedule(
            &mut base.systems.render_graph,
            base.context.as_ref(),
            &base.systems.descriptor_pool,
            &base.systems.pipeline_cache,
        );

        let renderer = self.renderer.get();

        let result_image = if let Some(renderer) = renderer {
            Some(renderer.render(
                &mut self.progress,
                &base.context,
                &mut schedule,
                &base.systems.pipeline_cache,
                &base.systems.descriptor_pool,
                &self.view_adjust.to_camera(),
            ))
        } else {
            None
        };

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

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new().with_color(swap_image, &[0f32, 0f32, 0f32, 0f32]);

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if let Some(result_image) = result_image {
                    params.add_image(result_image, ImageUsage::FRAGMENT_STORAGE_READ);
                }
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let pixels_per_point = base.egui_ctx.pixels_per_point();
                let egui_renderer = &mut base.egui_renderer;
                let show_debug_ui = self.show_debug_ui;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);

                    if let Some(result_image) = result_image {
                        let renderer_params = &renderer.unwrap().params;

                        let rec709_from_xyz = rec709_from_xyz_matrix()
                            * chromatic_adaptation_matrix(
                                bradford_lms_from_xyz_matrix(),
                                WhitePoint::D65,
                                renderer_params.observer_white_point(),
                            );
                        let acescg_from_xyz = ap1_from_xyz_matrix()
                            * chromatic_adaptation_matrix(
                                bradford_lms_from_xyz_matrix(),
                                WhitePoint::D60,
                                renderer_params.observer_white_point(),
                            );

                        let copy_descriptor_set = CopyDescriptorSet::create(
                            descriptor_pool,
                            |buf: &mut CopyData| {
                                *buf = CopyData {
                                    exposure_scale: renderer_params.log2_exposure_scale.exp2(),
                                    rec709_from_xyz,
                                    acescg_from_xyz,
                                    tone_map_method: renderer_params.tone_map_method.into_integer(),
                                }
                            },
                            params.get_image_view(result_image, ImageViewDesc::default()),
                        );

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count);

                        draw_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            &state,
                            "path_tracer/copy.vert.spv",
                            "path_tracer/copy.frag.spv",
                            copy_descriptor_set,
                            3,
                        );
                    }

                    // draw ui
                    if show_debug_ui {
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
    context: SharedContext,
    size: u32,
    mem: vk::DeviceMemory,
    buffer: UniqueBuffer,
    mapping: *const u8,
}

impl CaptureBuffer {
    fn new(context: &SharedContext, size: u32) -> Self {
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

        let mapped_memory_range = vk::MappedMemoryRange {
            memory: Some(mem),
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };
        unsafe {
            context
                .device
                .flush_mapped_memory_ranges(slice::from_ref(&mapped_memory_range))
        }
        .unwrap();

        Self {
            context: SharedContext::clone(context),
            size,
            mem,
            buffer: Unique::new(buffer, context.allocate_handle_uid()),
            mapping: mapping as *const _,
        }
    }

    fn mapping(&self) -> &[u8] {
        let mapped_memory_range = vk::MappedMemoryRange {
            memory: Some(self.mem),
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };
        unsafe {
            self.context
                .device
                .invalidate_mapped_memory_ranges(slice::from_ref(&mapped_memory_range))
        }
        .unwrap();

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
    context: SharedContext,
    systems: AppSystems,
    scene: SharedScene,
    renderer: TaskOutput<Renderer>,
    progress: RenderProgress,

    capture_buffer: CaptureBuffer,
}

impl CommandlineApp {
    fn new(context_params: &ContextParams, scene: Scene, renderer_params: RendererParams) -> Self {
        let context = Context::new(None, context_params);
        let systems = AppSystems::new(&context);

        let capture_buffer_size = renderer_params.width * renderer_params.height * 3;

        let scene = Arc::new(scene);
        let renderer = systems.task_system.spawn_task(Renderer::new(
            systems.resource_loader.clone(),
            Arc::clone(&scene),
            renderer_params,
        ));
        let progress = RenderProgress::new();

        let capture_buffer = CaptureBuffer::new(&context, capture_buffer_size);

        Self {
            context,
            systems,
            scene,
            renderer,
            progress,
            capture_buffer,
        }
    }

    fn run(&mut self, filename: &Path) {
        let mut start_instant = None;
        loop {
            let cbar = self.systems.acquire_command_buffer();

            let mut schedule = self.systems.resource_loader.begin_schedule(
                &mut self.systems.render_graph,
                self.context.as_ref(),
                &self.systems.descriptor_pool,
                &self.systems.pipeline_cache,
            );

            let mut copy_done = false;
            if let Some(renderer) = self.renderer.get() {
                let mut camera = *self.scene.cameras.first().unwrap();
                if let Some(fov_y_override) = renderer.params.fov_y_override {
                    *match &mut camera {
                        Camera::Pinhole { fov_y, .. } => fov_y,
                        Camera::ThinLens { fov_y, .. } => fov_y,
                    } = fov_y_override;
                }
                let result_image = renderer.render(
                    &mut self.progress,
                    &self.context,
                    &mut schedule,
                    &self.systems.pipeline_cache,
                    &self.systems.descriptor_pool,
                    &camera,
                );

                if start_instant.is_none() {
                    println!("starting render");
                    start_instant = Some(Instant::now());
                }

                if self.progress.done(&renderer.params) {
                    let capture_desc = BufferDesc::new(self.capture_buffer.size as usize);
                    let capture_buffer = schedule.import_buffer(
                        &capture_desc,
                        BufferUsage::COMPUTE_STORAGE_WRITE,
                        self.capture_buffer.buffer,
                        BufferUsage::empty(),
                        BufferUsage::empty(),
                    );

                    schedule.add_compute(
                        command_name!("capture"),
                        |params| {
                            params.add_image(result_image, ImageUsage::COMPUTE_STORAGE_READ);
                            params.add_buffer(capture_buffer, BufferUsage::COMPUTE_STORAGE_WRITE);
                        },
                        {
                            let pipeline_cache = &self.systems.pipeline_cache;
                            let descriptor_pool = &self.systems.descriptor_pool;
                            let context = &self.context;
                            let renderer_params = &renderer.params;
                            move |params, cmd| {
                                let rec709_from_xyz = rec709_from_xyz_matrix()
                                    * chromatic_adaptation_matrix(
                                        bradford_lms_from_xyz_matrix(),
                                        WhitePoint::D65,
                                        WhitePoint::E,
                                    );
                                let acescg_from_xyz = ap1_from_xyz_matrix()
                                    * chromatic_adaptation_matrix(
                                        bradford_lms_from_xyz_matrix(),
                                        WhitePoint::D60,
                                        WhitePoint::E,
                                    );

                                let descriptor_set = CaptureDescriptorSet::create(
                                    descriptor_pool,
                                    |buf: &mut CaptureData| {
                                        *buf = CaptureData {
                                            size: renderer_params.size(),
                                            exposure_scale: renderer_params.log2_exposure_scale.exp2(),
                                            rec709_from_xyz,
                                            acescg_from_xyz,
                                            tone_map_method: renderer_params.tone_map_method.into_integer(),
                                        };
                                    },
                                    params.get_buffer(capture_buffer),
                                    params.get_image_view(result_image, ImageViewDesc::default()),
                                );

                                dispatch_helper(
                                    &context.device,
                                    pipeline_cache,
                                    cmd,
                                    "path_tracer/capture.comp.spv",
                                    &[],
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

        println!("waiting for fence");
        self.systems.command_buffer_pool.wait_after_submit();
        println!(
            "render time for {} samples: {} seconds",
            self.renderer.get().unwrap().params.sample_count(),
            Instant::now().duration_since(start_instant.unwrap()).as_secs_f32()
        );

        println!("saving image to {:?}", filename);
        let params = &self.renderer.get().unwrap().params;
        match filename.extension().unwrap().to_str().unwrap() {
            "png" => {
                stb::image_write::stbi_write_png(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    params.width as i32,
                    params.height as i32,
                    3,
                    self.capture_buffer.mapping(),
                    (params.width * 3) as i32,
                )
                .unwrap();
            }
            "tga" => {
                stb::image_write::stbi_write_tga(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    params.width as i32,
                    params.height as i32,
                    3,
                    self.capture_buffer.mapping(),
                )
                .unwrap();
            }
            "jpg" => {
                let quality = 95;
                stb::image_write::stbi_write_jpg(
                    CString::new(filename.to_str().unwrap()).unwrap().as_c_str(),
                    params.width as i32,
                    params.height as i32,
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

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum MaterialTestVariant {
    Conductors,
    Gold,
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
    /// Material test scene
    MaterialTest {
        ply_filename: PathBuf,
        #[structopt(possible_values=&MaterialTestVariant::VARIANTS)]
        variant: MaterialTestVariant,
        illuminant: Illuminant,
    },
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

    /// Run fullscreen
    #[structopt(long, global = true)]
    fullscreen: bool,

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
        bindless: ContextFeature::Require,
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
        SceneDesc::Tungsten { filename } => tungsten::load_scene(filename, renderer_params.observer_illuminant()),
        SceneDesc::MaterialTest {
            ply_filename,
            variant,
            illuminant,
        } => {
            let surfaces: &[Surface] = match variant {
                MaterialTestVariant::Conductors => &[
                    Surface::RoughConductor {
                        conductor: Conductor::Gold,
                        roughness: 0.2,
                    },
                    Surface::RoughConductor {
                        conductor: Conductor::Iron,
                        roughness: 0.2,
                    },
                    Surface::RoughConductor {
                        conductor: Conductor::Copper,
                        roughness: 0.2,
                    },
                ],
                MaterialTestVariant::Gold => &[Surface::RoughConductor {
                    conductor: Conductor::Gold,
                    roughness: 0.2,
                }],
            };
            create_material_test_scene(ply_filename, surfaces, *illuminant)
        }
    };
    if scene.cameras.is_empty() {
        panic!("scene must contain at least one camera!");
    }

    if let Some(output) = app_params.output {
        let mut app = CommandlineApp::new(&context_params, scene, renderer_params);
        app.run(&output);
    } else {
        let event_loop = EventLoop::new();

        let mut window_builder = WindowBuilder::new().with_title("trace");
        window_builder = if app_params.fullscreen {
            let monitor = event_loop.primary_monitor().unwrap();
            let size = PhysicalSize::new(renderer_params.width, renderer_params.height);
            let video_mode = monitor
                .video_modes()
                .filter(|m| m.size() == size)
                .max_by(|a, b| {
                    let t = |m: &VideoMode| (m.bit_depth(), m.refresh_rate_millihertz());
                    Ord::cmp(&t(a), &t(b))
                })
                .unwrap();
            window_builder.with_fullscreen(Some(Fullscreen::Exclusive(video_mode)))
        } else {
            window_builder.with_inner_size(Size::Physical(PhysicalSize::new(
                renderer_params.width,
                renderer_params.height,
            )))
        };
        let window = window_builder.build(&event_loop).unwrap();

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
