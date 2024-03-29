use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use primes::{PrimeSet as _, Sieve};
use rand::{distributions::Uniform, prelude::*, rngs::SmallRng};
use spark::vk;
use std::mem;
use structopt::StructOpt;
use strum::VariantNames;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

struct Primes(Sieve);

impl Primes {
    fn new() -> Self {
        Self(Sieve::new())
    }

    fn next_after(&mut self, n: u32) -> u32 {
        self.0.find(n as u64).1 as u32
    }
}

const MAX_AGE: usize = 15;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct HashTableInfo {
    entry_count: u32,
    store_max_age: u32,
    offsets: [u32; MAX_AGE],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CircleParams {
    centre: Vec2,
    radius: f32,
}

const CIRCLE_COUNT: usize = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GenerateImageUniforms {
    circles: [CircleParams; CIRCLE_COUNT],
}

descriptor_set!(GenerateImageDescriptorSet {
    uniforms: UniformData<GenerateImageUniforms>,
    image: StorageImage
});

descriptor_set!(ClearHashTableDescriptorSet {
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
    max_ages: StorageBuffer,
    age_histogram: StorageBuffer,
});

descriptor_set!(UpdateHashTableDescriptorSet {
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
    max_ages: StorageBuffer,
    image: StorageImage,
});

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugQuadUniforms {
    ortho_from_quad: Scale2Offset2,
}

descriptor_set!(DebugImageDescriptorSet {
    debug_quad: UniformData<DebugQuadUniforms>,
    input_image: StorageImage,
    output_image: StorageImage
});

descriptor_set!(DebugHashTableDescriptorSet {
    debug_quad: UniformData<DebugQuadUniforms>,
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
});

descriptor_set!(MakeAgeHistogramDescriptorSet {
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
    age_histogram: StorageBuffer,
});

descriptor_set!(DebugAgeHistogramDescriptorSet {
    debug_quad: UniformData<DebugQuadUniforms>,
    hash_table_info: UniformData<HashTableInfo>,
    age_histogram: StorageBuffer,
});

struct App {
    rng: SmallRng,
    primes: Primes,
    store_max_age: bool,
    table_size: f32,
    circles: [CircleParams; CIRCLE_COUNT],
    hash_table_offsets: [u32; MAX_AGE],
}

fn make_circles(rng: &mut SmallRng) -> [CircleParams; CIRCLE_COUNT] {
    let mut circles = ArrayVec::new();
    let dist = Uniform::new(0.0, 1.0);
    for _ in 0..CIRCLE_COUNT {
        circles.push(CircleParams {
            centre: Vec2::new(rng.sample(dist), rng.sample(dist)) * 1024.0,
            radius: rng.sample(dist) * 512.0,
        });
    }
    circles.into_inner().unwrap()
}

fn make_hash_table_offsets(rng: &mut SmallRng, primes: &mut Primes) -> [u32; MAX_AGE] {
    let dist = Uniform::new(1_000_000, 10_000_000);
    let mut offsets = ArrayVec::new();
    offsets.push(0);
    for _ in 1..MAX_AGE {
        let random_not_prime = rng.sample(dist);
        offsets.push(primes.next_after(random_not_prime));
    }
    offsets.into_inner().unwrap()
}

impl App {
    fn new(_base: &mut AppBase) -> Self {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut primes = Primes::new();
        let circles = make_circles(&mut rng);
        let hash_table_offsets = make_hash_table_offsets(&mut rng, &mut primes);
        println!("{:?}", hash_table_offsets);

        Self {
            rng,
            primes,
            store_max_age: true,
            table_size: 0.05,
            circles,
            hash_table_offsets,
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
                if ui.button("Random circles").clicked() {
                    self.circles = make_circles(&mut self.rng);
                }
                ui.add(egui::Slider::new(&mut self.table_size, 0.001..=0.12).prefix("Table size: "));
                ui.checkbox(&mut self.store_max_age, "Store max age");
                if ui.button("Random offsets").clicked() {
                    self.hash_table_offsets = make_hash_table_offsets(&mut self.rng, &mut self.primes);
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

        let main_sample_count = vk::SampleCountFlags::N1;
        let main_render_state = RenderState::new().with_color(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]);

        let image_size = UVec2::new(1024, 1024);
        let image_desc = ImageDesc::new_2d(image_size, vk::Format::R8_UNORM, vk::ImageAspectFlags::COLOR);
        let input_image = schedule.describe_image(&image_desc);
        let output_image = schedule.describe_image(&image_desc);

        let entry_count = ((1024.0 * 1024.0 * self.table_size) as u32) | 1;
        let hash_table_info = HashTableInfo {
            entry_count,
            store_max_age: if self.store_max_age { 1 } else { 0 },
            offsets: self.hash_table_offsets,
        };

        let buffer_desc = BufferDesc::new((entry_count as usize) * mem::size_of::<u32>());
        let entries_buffer_id = schedule.describe_buffer(&buffer_desc);
        let max_ages_buffer_id = schedule.describe_buffer(&buffer_desc);
        let age_histogram_buffer_id = schedule.describe_buffer(&BufferDesc::new(MAX_AGE * mem::size_of::<u32>()));

        schedule.add_compute(
            command_name!("generate_image"),
            |params| {
                params.add_image(input_image, ImageUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let circles = &self.circles;
                move |params, cmd| {
                    let descriptor_set = GenerateImageDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut GenerateImageUniforms| *buf = GenerateImageUniforms { circles: *circles },
                        params.get_image_view(input_image, ImageViewDesc::default()),
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        "coherent_hashing/generate_image.comp.spv",
                        &[],
                        descriptor_set,
                        image_size.div_round_up(16),
                    );
                }
            },
        );

        schedule.add_compute(
            command_name!("clear_hash_table"),
            |params| {
                params.add_buffer(entries_buffer_id, BufferUsage::COMPUTE_STORAGE_WRITE);
                params.add_buffer(max_ages_buffer_id, BufferUsage::COMPUTE_STORAGE_WRITE);
                params.add_buffer(age_histogram_buffer_id, BufferUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let descriptor_set = ClearHashTableDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        params.get_buffer(entries_buffer_id),
                        params.get_buffer(max_ages_buffer_id),
                        params.get_buffer(age_histogram_buffer_id),
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        "coherent_hashing/clear_hash_table.comp.spv",
                        &[],
                        descriptor_set,
                        UVec2::new(entry_count.div_round_up(64), 1),
                    );
                }
            },
        );

        schedule.add_compute(
            command_name!("write_hash_table"),
            |params| {
                params.add_buffer(entries_buffer_id, BufferUsage::COMPUTE_STORAGE_ATOMIC);
                params.add_buffer(max_ages_buffer_id, BufferUsage::COMPUTE_STORAGE_ATOMIC);
                params.add_image(input_image, ImageUsage::COMPUTE_STORAGE_READ);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let descriptor_set = UpdateHashTableDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        params.get_buffer(entries_buffer_id),
                        params.get_buffer(max_ages_buffer_id),
                        params.get_image_view(input_image, ImageViewDesc::default()),
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        "coherent_hashing/write_hash_table.comp.spv",
                        &[],
                        descriptor_set,
                        image_size.div_round_up(16),
                    );
                }
            },
        );

        schedule.add_compute(
            command_name!("make_age_histogram"),
            |params| {
                params.add_buffer(entries_buffer_id, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_buffer(age_histogram_buffer_id, BufferUsage::COMPUTE_STORAGE_ATOMIC);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let descriptor_set = MakeAgeHistogramDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        params.get_buffer(entries_buffer_id),
                        params.get_buffer(age_histogram_buffer_id),
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        "coherent_hashing/make_age_histogram.comp.spv",
                        &[],
                        descriptor_set,
                        UVec2::new(entry_count.div_round_up(64), 1),
                    );
                }
            },
        );

        schedule.add_compute(
            command_name!("read_hash_table"),
            |params| {
                params.add_buffer(entries_buffer_id, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_buffer(max_ages_buffer_id, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_image(output_image, ImageUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let descriptor_set = UpdateHashTableDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        params.get_buffer(entries_buffer_id),
                        params.get_buffer(max_ages_buffer_id),
                        params.get_image_view(output_image, ImageViewDesc::default()),
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        "coherent_hashing/read_hash_table.comp.spv",
                        &[],
                        descriptor_set,
                        image_size.div_round_up(16),
                    );
                }
            },
        );

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                params.add_image(input_image, ImageUsage::FRAGMENT_STORAGE_READ);
                params.add_image(output_image, ImageUsage::FRAGMENT_STORAGE_READ);
                params.add_buffer(entries_buffer_id, BufferUsage::FRAGMENT_STORAGE_READ);
                params.add_buffer(age_histogram_buffer_id, BufferUsage::FRAGMENT_STORAGE_READ);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let pixels_per_point = base.egui_ctx.pixels_per_point();
                let egui_renderer = &mut base.egui_renderer;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_size);
                    let ortho_from_screen =
                        Scale2Offset2::new(Vec2::broadcast(2.0) / swap_size.as_float(), Vec2::broadcast(-1.0));

                    // visualise results
                    let screen_from_image = Scale2Offset2::new(image_size.as_float(), Vec2::broadcast(10.0));
                    let descriptor_set = DebugImageDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_image,
                            };
                        },
                        params.get_image_view(input_image, ImageViewDesc::default()),
                        params.get_image_view(output_image, ImageViewDesc::default()),
                    );
                    let state = GraphicsPipelineState::new(render_pass, main_sample_count)
                        .with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_image.frag.spv",
                        descriptor_set,
                        4,
                    );

                    let screen_from_table = Scale2Offset2::new(Vec2::new(128.0, 1024.0), Vec2::new(1044.0, 10.0));
                    let descriptor_set = DebugHashTableDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_table,
                            };
                        },
                        |buf: &mut HashTableInfo| {
                            *buf = hash_table_info;
                        },
                        params.get_buffer(entries_buffer_id),
                    );
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_hash_table.frag.spv",
                        descriptor_set,
                        4,
                    );

                    let screen_from_histogram = Scale2Offset2::new(Vec2::new(160.0, 80.0), Vec2::new(1184.0, 10.0));
                    let descriptor_set = DebugAgeHistogramDescriptorSet::create(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_histogram,
                            };
                        },
                        |buf: &mut HashTableInfo| {
                            *buf = hash_table_info;
                        },
                        params.get_buffer(age_histogram_buffer_id),
                    );
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_age_histogram.frag.spv",
                        descriptor_set,
                        4,
                    );

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
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        ..Default::default()
    };

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("coherent_hashing")
        .with_inner_size(Size::Logical(LogicalSize::new(1354.0, 1044.0)))
        .build(&event_loop)
        .unwrap();

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
