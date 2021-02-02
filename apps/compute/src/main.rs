use bytemuck::{Pod, Zeroable};
use caldera::*;
use imgui::im_str;
use imgui::{Drag, Key, Slider};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use spark::vk;
use std::env;
use std::sync::Arc;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    monitor::VideoMode,
    window::{Fullscreen, WindowBuilder},
};

mod color_space;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SamplePixel {
    x: u16,
    y: u16,
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct TraceData {
    dims: UVec2,
    dims_rcp: Vec2,
    pass_index: u32,
}

descriptor_set_layout!(TraceDescriptorSetLayout {
    trace: UniformData<TraceData>,
    result_r: StorageImage,
    result_g: StorageImage,
    result_b: StorageImage,
    samples: StorageImage,
});

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct CopyData {
    offset: IVec2,
    trace_dims: UVec2,
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
    const SEQUENCE_COUNT: u32 = 4096;
    const SAMPLES_PER_SEQUENCE: u32 = 256;
    const MAX_PASS_COUNT: u32 = Self::SAMPLES_PER_SEQUENCE / 4;

    fn trace_image_size() -> UVec2 {
        UVec2::new(640, 480)
    }

    fn new(base: &mut AppBase) -> Self {
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let trace_descriptor_set_layout = TraceDescriptorSetLayout::new(descriptor_set_layout_cache);
        let trace_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(trace_descriptor_set_layout.0);

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
                Self::SAMPLES_PER_SEQUENCE,
                Self::SEQUENCE_COUNT,
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

        let trace_images = {
            let size = Self::trace_image_size();
            let desc = ImageDesc::new_2d(size.x, size.y, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR);
            let usage = ImageUsage::FRAGMENT_STORAGE_READ
                | ImageUsage::COMPUTE_STORAGE_READ
                | ImageUsage::COMPUTE_STORAGE_WRITE;
            let render_graph = &mut base.systems.render_graph;
            let global_allocator = &mut base.systems.global_allocator;
            (
                render_graph.create_image(&desc, usage, global_allocator),
                render_graph.create_image(&desc, usage, global_allocator),
                render_graph.create_image(&desc, usage, global_allocator),
            )
        };

        Self {
            context: Arc::clone(&base.context),

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
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let sample_image_view = base.systems.resource_loader.get_image_view(self.sample_image);

        let trace_image_size = Self::trace_image_size();
        let pass_count = if let Some(sample_image_view) =
            sample_image_view.filter(|_| self.next_pass_index != self.target_pass_count)
        {
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
                {
                    let context = &base.context;
                    let descriptor_pool = &base.systems.descriptor_pool;
                    let pipeline_cache = &base.systems.pipeline_cache;
                    let trace_images = &self.trace_images;
                    let trace_descriptor_set_layout = &self.trace_descriptor_set_layout;
                    let trace_pipeline_layout = self.trace_pipeline_layout;
                    let next_pass_index = self.next_pass_index;
                    move |params, cmd| {
                        let sample_image_view = sample_image_view;
                        let trace_image_views = (
                            params.get_image_view(trace_images.0),
                            params.get_image_view(trace_images.1),
                            params.get_image_view(trace_images.2),
                        );

                        let descriptor_set = trace_descriptor_set_layout.write(
                            &descriptor_pool,
                            |buf: &mut TraceData| {
                                let dims_rcp = Vec2::broadcast(1.0) / trace_image_size.as_float();
                                *buf = TraceData {
                                    dims: trace_image_size.into(),
                                    dims_rcp: dims_rcp.into(),
                                    pass_index: next_pass_index,
                                };
                            },
                            trace_image_views.0,
                            trace_image_views.1,
                            trace_image_views.2,
                            sample_image_view,
                        );

                        dispatch_helper(
                            &context.device,
                            &pipeline_cache,
                            cmd,
                            trace_pipeline_layout,
                            "compute/trace.comp.spv",
                            descriptor_set,
                            trace_image_size.div_round_up(16),
                        );
                    }
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
                params.add_image(self.trace_images.0, ImageUsage::FRAGMENT_STORAGE_READ);
                params.add_image(self.trace_images.1, ImageUsage::FRAGMENT_STORAGE_READ);
                params.add_image(self.trace_images.2, ImageUsage::FRAGMENT_STORAGE_READ);
            },
            {
                let context = &base.context;
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let trace_images = &self.trace_images;
                let copy_descriptor_set_layout = &self.copy_descriptor_set_layout;
                let log2_exposure_scale = self.log2_exposure_scale;
                let copy_pipeline_layout = self.copy_pipeline_layout;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    let trace_image_views = (
                        params.get_image_view(trace_images.0),
                        params.get_image_view(trace_images.1),
                        params.get_image_view(trace_images.2),
                    );
                    let swap_size = UVec2::new(swap_extent.width, swap_extent.height);

                    set_viewport_helper(&context.device, cmd, swap_extent);

                    let copy_descriptor_set = copy_descriptor_set_layout.write(
                        &descriptor_pool,
                        |buf| {
                            *buf = CopyData {
                                offset: ((trace_image_size.as_signed() - swap_size.as_signed()) / 2).into(),
                                trace_dims: trace_image_size.into(),
                                trace_scale: log2_exposure_scale.exp2() / (pass_count as f32),
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
                        copy_pipeline_layout,
                        &GraphicsPipelineState::new(render_pass, main_sample_count),
                        "compute/copy.vert.spv",
                        "compute/copy.frag.spv",
                        copy_descriptor_set,
                        3,
                    );

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

        self.next_pass_index = pass_count;
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
