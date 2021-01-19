use caldera::*;
use caldera_macro::descriptor_set_layout;
use imgui::im_str;
use imgui::{Drag, Key, Slider};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use spark::vk;
use std::env;
use std::ffi::CStr;
use std::sync::Arc;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    monitor::VideoMode,
    window::{Fullscreen, WindowBuilder},
};

mod color_space;

#[derive(Clone, Copy)]
#[repr(C)]
struct TraceData {
    dims: [u32; 2],
    dims_rcp: [f32; 2],
    pass_index: u32,
}

descriptor_set_layout!(TraceDescriptorSetLayout {
    trace: UniformData<TraceData>,
    result_r: StorageImage,
    result_g: StorageImage,
    result_b: StorageImage,
    samples: StorageImage,
});

#[derive(Clone, Copy)]
#[repr(C)]
struct CopyData {
    offset: [i32; 2],
    trace_dims: [u32; 2],
    trace_scale: f32,
}

descriptor_set_layout!(CopyDescriptorSetLayout {
    copy: UniformData<CopyData>,
    image_r: StorageImage,
    image_g: StorageImage,
    image_b: StorageImage,
});

struct App {
    context: Arc<Context>,

    trace_descriptor_set_layout: TraceDescriptorSetLayout,
    trace_pipeline_layout: vk::PipelineLayout,
    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    sample_image: StaticImageHandle,
    trace_images: (ImageHandle, ImageHandle, ImageHandle),

    log2_exposure_scale: f32,
    target_pass_count: u32,
    next_pass_index: u32,
}

impl App {
    const SAMPLES_PER_SEQUENCE: u32 = 256;
    const MAX_PASS_COUNT: u32 = Self::SAMPLES_PER_SEQUENCE / 4;

    fn trace_image_size() -> UVec2 {
        UVec2::new(640, 480)
    }

    fn new(base: &mut AppBase) -> Self {
        let context = &base.context;
        let descriptor_pool = &base.systems.descriptor_pool;
        let global_allocator = &mut base.systems.global_allocator;
        let resource_loader = &mut base.systems.resource_loader;

        let trace_descriptor_set_layout = TraceDescriptorSetLayout::new(&descriptor_pool);
        let trace_pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout_from_ref(&trace_descriptor_set_layout.0)
        }
        .unwrap();

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(&descriptor_pool);
        let copy_pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout_from_ref(&copy_descriptor_set_layout.0)
        }
        .unwrap();

        let sample_image = resource_loader.create_image();
        resource_loader.async_load(move |allocator| {
            const TILE_SIZE: u32 = 64;
            const PIXEL_COUNT: usize = (TILE_SIZE * TILE_SIZE) as usize;

            let sequences: Vec<Vec<_>> = (0..PIXEL_COUNT)
                .into_par_iter()
                .map(|i| {
                    let mut rng = SmallRng::seed_from_u64(i as u64);
                    pmj::generate(Self::SAMPLES_PER_SEQUENCE as usize, 4, &mut rng)
                })
                .collect();

            let desc = ImageDesc::new_2d(
                TILE_SIZE,
                TILE_SIZE,
                vk::Format::R16G16_UINT,
                vk::ImageAspectFlags::COLOR,
            )
            .with_layer_count(Self::SAMPLES_PER_SEQUENCE as u32);

            let mut mapping = allocator
                .map_image::<u16>(sample_image, &desc, ImageUsage::COMPUTE_STORAGE_READ)
                .unwrap();

            let buffer = mapping.get_mut();
            let mut write_offset = 0;
            for sample_index in 0..Self::SAMPLES_PER_SEQUENCE {
                for sequence_index in 0..PIXEL_COUNT {
                    let sample = sequences[sequence_index][sample_index as usize];
                    buffer[write_offset + 0] = sample.x_bits(16) as u16;
                    buffer[write_offset + 1] = sample.y_bits(16) as u16;
                    write_offset += 2;
                }
            }
        });

        let trace_images = {
            let render_graph = &mut base.systems.render_graph;
            let trace_image_size = Self::trace_image_size();
            let mut trace_image_alloc = || -> ImageHandle {
                render_graph.create_image(
                    &ImageDesc::new_2d(
                        trace_image_size.x,
                        trace_image_size.y,
                        vk::Format::R32_SFLOAT,
                        vk::ImageAspectFlags::COLOR,
                    ),
                    ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::COMPUTE_STORAGE_WRITE,
                    global_allocator,
                )
            };
            (trace_image_alloc(), trace_image_alloc(), trace_image_alloc())
        };

        Self {
            context: Arc::clone(&context),

            trace_descriptor_set_layout,
            trace_pipeline_layout,
            copy_descriptor_set_layout,
            copy_pipeline_layout,

            sample_image,
            trace_images,

            log2_exposure_scale: 0f32,
            target_pass_count: 16,
            next_pass_index: 0,
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
            .build(&ui, || {
                Slider::new(im_str!("Target Pass Count"))
                    .range(1..=Self::MAX_PASS_COUNT)
                    .build(&ui, &mut self.target_pass_count);
                ui.text(format!("Passes: {}", self.next_pass_index));
                if ui.button(im_str!("Reset"), [0.0, 0.0]) {
                    self.next_pass_index = 0;
                }
                Drag::new(im_str!("Exposure"))
                    .speed(0.05f32)
                    .build(&ui, &mut self.log2_exposure_scale);
            });

        let cbar = base.systems.acquire_command_buffer();
        let ui_platform = &mut base.ui_platform;
        let ui_renderer = &mut base.ui_renderer;
        let window = &base.window;
        ui_renderer.begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut ui = Some(ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let context = &base.context;
        let descriptor_pool = &base.systems.descriptor_pool;
        let pipeline_cache = &base.systems.pipeline_cache;
        let resource_loader = &base.systems.resource_loader;

        let sample_image_view = resource_loader.get_image_view(self.sample_image);

        let trace_image_size = Self::trace_image_size();
        let pass_count = if self.next_pass_index != self.target_pass_count && sample_image_view.is_some() {
            if self.next_pass_index > self.target_pass_count {
                self.next_pass_index = 0;
            }
            schedule.add_compute(
                command_name!("trace"),
                |params| {
                    params.add_image(
                        self.trace_images.0,
                        ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::COMPUTE_STORAGE_WRITE,
                    );
                    params.add_image(
                        self.trace_images.1,
                        ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::COMPUTE_STORAGE_WRITE,
                    );
                    params.add_image(
                        self.trace_images.2,
                        ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::COMPUTE_STORAGE_WRITE,
                    );
                },
                |params, cmd| {
                    let trace_image_views = (
                        params.get_image_view(self.trace_images.0),
                        params.get_image_view(self.trace_images.1),
                        params.get_image_view(self.trace_images.2),
                    );

                    let descriptor_set = self.trace_descriptor_set_layout.write(
                        &descriptor_pool,
                        &|buf: &mut TraceData| {
                            let dims_rcp = Vec2::broadcast(1.0) / trace_image_size.as_float();
                            *buf = TraceData {
                                dims: trace_image_size.into(),
                                dims_rcp: dims_rcp.into(),
                                pass_index: self.next_pass_index,
                            };
                        },
                        trace_image_views.0,
                        trace_image_views.1,
                        trace_image_views.2,
                        sample_image_view.unwrap(),
                    );

                    dispatch_helper(
                        &context.device,
                        &pipeline_cache,
                        cmd,
                        self.trace_pipeline_layout,
                        "compute/trace.comp.spv",
                        descriptor_set,
                        trace_image_size.div_round_up(16),
                    );
                },
            );
            self.next_pass_index + 1
        } else {
            self.next_pass_index
        };

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

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new(swap_image, &[0f32, 0f32, 0f32, 0f32]);

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                params.add_image(self.trace_images.0, ImageUsage::COMPUTE_STORAGE_READ);
                params.add_image(self.trace_images.1, ImageUsage::COMPUTE_STORAGE_READ);
                params.add_image(self.trace_images.2, ImageUsage::COMPUTE_STORAGE_READ);
            },
            |params, cmd, render_pass| {
                let trace_image_views = (
                    params.get_image_view(self.trace_images.0),
                    params.get_image_view(self.trace_images.1),
                    params.get_image_view(self.trace_images.2),
                );
                let swap_size = UVec2::new(swap_extent.width, swap_extent.height);

                set_viewport_helper(&context.device, cmd, swap_extent);

                let copy_descriptor_set = self.copy_descriptor_set_layout.write(
                    &descriptor_pool,
                    &|buf| {
                        *buf = CopyData {
                            offset: ((trace_image_size.as_signed() - swap_size.as_signed()) / 2).into(),
                            trace_dims: trace_image_size.into(),
                            trace_scale: self.log2_exposure_scale.exp2() / (pass_count as f32),
                        };
                    },
                    trace_image_views.0,
                    trace_image_views.1,
                    trace_image_views.2,
                );
                draw_helper(
                    &context.device,
                    &pipeline_cache,
                    cmd,
                    self.copy_pipeline_layout,
                    &GraphicsPipelineState::new(render_pass, main_sample_count),
                    "compute/copy.vert.spv",
                    "compute/copy.frag.spv",
                    copy_descriptor_set,
                    3,
                );

                if let Some(ui) = ui.take() {
                    ui_platform.prepare_render(&ui, window);

                    let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                    ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
                }
            },
        );

        schedule.run(
            &context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            swap_image,
            &mut base.systems.query_pool,
        );
        drop(ui);

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display.present(swap_vk_image, rendering_finished_semaphore);

        self.next_pass_index = pass_count;
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let device = self.context.device;
        unsafe {
            device.destroy_descriptor_set_layout(Some(self.trace_descriptor_set_layout.0), None);
            device.destroy_descriptor_set_layout(Some(self.copy_descriptor_set_layout.0), None);

            device.destroy_pipeline_layout(Some(self.trace_pipeline_layout), None);
            device.destroy_pipeline_layout(Some(self.copy_pipeline_layout), None);
        }
    }
}

fn main() {
    let mut params = ContextParams::default();
    let mut is_fullscreen = false;
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        match arg {
            "-f" => is_fullscreen = true,
            "--test" => {
                color_space::derive_matrices();
                return;
            }
            _ => {
                if !params.parse_arg(arg) {
                    panic!("unknown argument {:?}", arg);
                }
            }
        }
    }

    let event_loop = EventLoop::new();

    let mut window_builder = WindowBuilder::new().with_title("compute");
    window_builder = if is_fullscreen {
        let monitor = event_loop.primary_monitor().unwrap();
        let size = monitor.size();
        let video_mode = monitor
            .video_modes()
            .filter(|m| m.size() == size)
            .max_by(|a, b| {
                let t = |m: &VideoMode| (m.bit_depth(), m.refresh_rate());
                Ord::cmp(&t(a), &t(b))
            })
            .unwrap();
        println!(
            "full screen mode: {}x{} {}bpp {}Hz",
            video_mode.size().width,
            video_mode.size().height,
            video_mode.bit_depth(),
            video_mode.refresh_rate()
        );
        window_builder.with_fullscreen(Some(Fullscreen::Exclusive(video_mode)))
    } else {
        window_builder.with_inner_size(Size::Logical(LogicalSize::new(640.0, 480.0)))
    };
    let window = window_builder.build(&event_loop).unwrap();

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
