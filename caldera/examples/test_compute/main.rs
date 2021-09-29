use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use imgui::{Drag, Key, Slider};
use rand::{prelude::*, rngs::SmallRng};
use rayon::prelude::*;
use spark::vk;
use structopt::StructOpt;
use strum::VariantNames;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    monitor::VideoMode,
    window::{Fullscreen, WindowBuilder},
};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct TraceData {
    dims: UVec2,
    dims_rcp: Vec2,
    pass_index: u32,
}

descriptor_set_layout!(TraceDescriptorSet {
    trace: UniformData<TraceData>,
    result: [StorageImage; 3],
    samples: StorageImage,
});

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct CopyData {
    offset: IVec2,
    trace_dims: UVec2,
    trace_scale: f32,
}

descriptor_set_layout!(CopyDescriptorSet {
    copy: UniformData<CopyData>,
    image: [StorageImage; 3],
});

struct App {
    context: SharedContext,

    sample_image: StaticImageHandle,
    trace_images: (ImageHandle, ImageHandle, ImageHandle),

    log2_exposure_scale: f32,
    target_pass_count: u32,
    next_pass_index: u32,
}

impl App {
    const SEQUENCE_COUNT: u32 = 1024;
    const SAMPLES_PER_SEQUENCE: u32 = 256;
    const MAX_PASS_COUNT: u32 = Self::SAMPLES_PER_SEQUENCE / 4;

    fn trace_image_size() -> UVec2 {
        UVec2::new(640, 480)
    }

    fn new(base: &mut AppBase) -> Self {
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
                vk::Format::R32G32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = allocator
                .map_image(sample_image, &desc, ImageUsage::COMPUTE_STORAGE_READ)
                .unwrap();

            for sample in sequences.iter().flat_map(|sequence| sequence.iter()) {
                let pixel: [f32; 2] = [sample.x(), sample.y()];
                writer.write(&pixel);
            }
        });

        let trace_images = {
            let size = Self::trace_image_size();
            let desc = ImageDesc::new_2d(size, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR);
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
            context: SharedContext::clone(&base.context),

            sample_image,
            trace_images,

            log2_exposure_scale: 0f32,
            target_pass_count: 16,
            next_pass_index: 0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(Key::Escape) {
            base.exit_requested = true;
        }
        imgui::Window::new("Debug")
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, || {
                let mut needs_reset = false;
                Drag::new("Exposure")
                    .speed(0.05f32)
                    .build(&ui, &mut self.log2_exposure_scale);
                Slider::new("Target Pass Count", 1, Self::MAX_PASS_COUNT).build(&ui, &mut self.target_pass_count);
                ui.text(format!("Passes: {}", self.next_pass_index));
                needs_reset |= ui.button("Reset");

                if needs_reset {
                    self.next_pass_index = 0;
                }
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
                    let context = base.context.as_ref();
                    let descriptor_pool = &base.systems.descriptor_pool;
                    let pipeline_cache = &base.systems.pipeline_cache;
                    let trace_images = &self.trace_images;
                    let next_pass_index = self.next_pass_index;
                    move |params, cmd| {
                        let sample_image_view = sample_image_view;
                        let trace_image_views = [
                            params.get_image_view(trace_images.0),
                            params.get_image_view(trace_images.1),
                            params.get_image_view(trace_images.2),
                        ];

                        let descriptor_set = TraceDescriptorSet::create(
                            descriptor_pool,
                            |buf: &mut TraceData| {
                                let dims_rcp = Vec2::broadcast(1.0) / trace_image_size.as_float();
                                *buf = TraceData {
                                    dims: trace_image_size,
                                    dims_rcp,
                                    pass_index: next_pass_index,
                                };
                            },
                            &trace_image_views,
                            sample_image_view,
                        );

                        dispatch_helper2(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            "test_compute/trace.comp.spv",
                            &[],
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

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore.unwrap());
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
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let trace_images = &self.trace_images;
                let log2_exposure_scale = self.log2_exposure_scale;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    let trace_image_views = [
                        params.get_image_view(trace_images.0),
                        params.get_image_view(trace_images.1),
                        params.get_image_view(trace_images.2),
                    ];

                    set_viewport_helper(&context.device, cmd, swap_size);

                    let copy_descriptor_set = CopyDescriptorSet::create(
                        descriptor_pool,
                        |buf| {
                            *buf = CopyData {
                                offset: ((trace_image_size.as_signed() - swap_size.as_signed()) / 2),
                                trace_dims: trace_image_size,
                                trace_scale: log2_exposure_scale.exp2() / (pass_count as f32),
                            };
                        },
                        &trace_image_views,
                    );
                    draw_helper2(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        &GraphicsPipelineState::new(render_pass, main_sample_count),
                        "test_compute/copy.vert.spv",
                        "test_compute/copy.frag.spv",
                        copy_descriptor_set,
                        3,
                    );

                    // draw imgui
                    ui_platform.prepare_render(&ui, window);

                    let pipeline = pipeline_cache.get_ui(ui_renderer, render_pass, main_sample_count);
                    ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
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

        self.next_pass_index = pass_count;
    }
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    /// Core Vulkan version to load
    #[structopt(short, long, parse(try_from_str=try_version_from_str), default_value="1.0")]
    version: vk::Version,

    /// Whether to use EXT_inline_uniform_block
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional")]
    inline_uniform_block: ContextFeature,

    /// Run fullscreen
    #[structopt(short, long)]
    fullscreen: bool,

    /// Test ACES fit matrices and exit
    #[structopt(long)]
    test: bool,
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        ..Default::default()
    };
    if app_params.test {
        derive_aces_fit_matrices();
        return;
    }

    let event_loop = EventLoop::new();

    let mut window_builder = WindowBuilder::new().with_title("compute");
    window_builder = if app_params.fullscreen {
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

    let mut base = AppBase::new(window, &context_params);
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
