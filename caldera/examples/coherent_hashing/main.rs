use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use imgui::{Key, Slider};
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

descriptor_set_layout!(GenerateImageDescriptorSetLayout { image: StorageImage });

descriptor_set_layout!(ClearHashTableDescriptorSetLayout {
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
    max_ages: StorageBuffer,
    age_histogram: StorageBuffer,
});

descriptor_set_layout!(UpdateHashTableDescriptorSetLayout {
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

descriptor_set_layout!(DebugImageDescriptorSetLayout {
    debug_quad: UniformData<DebugQuadUniforms>,
    input_image: StorageImage,
    output_image: StorageImage
});

descriptor_set_layout!(DebugHashTableDescriptorSetLayout {
    debug_quad: UniformData<DebugQuadUniforms>,
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
});

descriptor_set_layout!(MakeAgeHistogramDescriptorSetLayout {
    hash_table_info: UniformData<HashTableInfo>,
    entries: StorageBuffer,
    age_histogram: StorageBuffer,
});

descriptor_set_layout!(DebugAgeHistogramDescriptorSetLayout {
    debug_quad: UniformData<DebugQuadUniforms>,
    hash_table_info: UniformData<HashTableInfo>,
    age_histogram: StorageBuffer,
});

struct App {
    context: SharedContext,

    generate_image_descriptor_set_layout: GenerateImageDescriptorSetLayout,
    generate_image_pipeline_layout: vk::PipelineLayout,
    clear_hash_table_descriptor_set_layout: ClearHashTableDescriptorSetLayout,
    clear_hash_table_pipeline_layout: vk::PipelineLayout,
    update_hash_table_descriptor_set_layout: UpdateHashTableDescriptorSetLayout,
    update_hash_table_pipeline_layout: vk::PipelineLayout,
    debug_image_descriptor_set_layout: DebugImageDescriptorSetLayout,
    debug_image_pipeline_layout: vk::PipelineLayout,
    debug_hash_table_descriptor_set_layout: DebugHashTableDescriptorSetLayout,
    debug_hash_table_pipeline_layout: vk::PipelineLayout,
    make_age_histogram_descriptor_set_layout: MakeAgeHistogramDescriptorSetLayout,
    make_age_histogram_pipeline_layout: vk::PipelineLayout,
    debug_age_histogram_descriptor_set_layout: DebugAgeHistogramDescriptorSetLayout,
    debug_age_histogram_pipeline_layout: vk::PipelineLayout,

    rng: SmallRng,
    primes: Primes,
    store_max_age: bool,
    table_size: f32,
    hash_table_offsets: [u32; MAX_AGE],
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
    fn new(base: &mut AppBase) -> Self {
        let context = SharedContext::clone(&base.context);
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let generate_image_descriptor_set_layout = GenerateImageDescriptorSetLayout::new(descriptor_set_layout_cache);
        let generate_image_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(generate_image_descriptor_set_layout.0);

        let clear_hash_table_descriptor_set_layout =
            ClearHashTableDescriptorSetLayout::new(descriptor_set_layout_cache);
        let clear_hash_table_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(clear_hash_table_descriptor_set_layout.0);

        let update_hash_table_descriptor_set_layout =
            UpdateHashTableDescriptorSetLayout::new(descriptor_set_layout_cache);
        let update_hash_table_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(update_hash_table_descriptor_set_layout.0);

        let debug_image_descriptor_set_layout = DebugImageDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_image_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(debug_image_descriptor_set_layout.0);

        let debug_hash_table_descriptor_set_layout =
            DebugHashTableDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_hash_table_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(debug_hash_table_descriptor_set_layout.0);

        let make_age_histogram_descriptor_set_layout =
            MakeAgeHistogramDescriptorSetLayout::new(descriptor_set_layout_cache);
        let make_age_histogram_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(make_age_histogram_descriptor_set_layout.0);

        let debug_age_histogram_descriptor_set_layout =
            DebugAgeHistogramDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_age_histogram_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(debug_age_histogram_descriptor_set_layout.0);

        let mut rng = SmallRng::seed_from_u64(0);
        let mut primes = Primes::new();
        let hash_table_offsets = make_hash_table_offsets(&mut rng, &mut primes);
        println!("{:?}", hash_table_offsets);

        Self {
            context,
            generate_image_descriptor_set_layout,
            generate_image_pipeline_layout,
            clear_hash_table_descriptor_set_layout,
            clear_hash_table_pipeline_layout,
            update_hash_table_descriptor_set_layout,
            update_hash_table_pipeline_layout,
            debug_image_descriptor_set_layout,
            debug_image_pipeline_layout,
            debug_hash_table_descriptor_set_layout,
            debug_hash_table_pipeline_layout,
            make_age_histogram_descriptor_set_layout,
            make_age_histogram_pipeline_layout,
            debug_age_histogram_descriptor_set_layout,
            debug_age_histogram_pipeline_layout,
            rng,
            primes,
            store_max_age: true,
            table_size: 0.05,
            hash_table_offsets,
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
                Slider::new("Table size", 0.001, 0.12).build(&ui, &mut self.table_size);
                ui.checkbox("Store max age", &mut self.store_max_age);
                if ui.button("Random offsets") {
                    self.hash_table_offsets = make_hash_table_offsets(&mut self.rng, &mut self.primes);
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

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
        let main_render_state = RenderState::new(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]);

        let image_size = UVec2::new(1024, 1024);
        let image_desc = ImageDesc::new_2d(image_size, vk::Format::R8_UNORM, vk::ImageAspectFlags::COLOR);
        let input_image = schedule.describe_image(&image_desc);
        let output_image = schedule.describe_image(&image_desc);

        let entry_count = self.primes.next_after((1024.0 * 1024.0 * self.table_size) as u32);
        let hash_table_info = HashTableInfo {
            entry_count,
            store_max_age: if self.store_max_age { 1 } else { 0 },
            offsets: self.hash_table_offsets,
        };

        let buffer_desc = BufferDesc::new((entry_count as usize) * mem::size_of::<u32>());
        let entries_buffer = schedule.describe_buffer(&buffer_desc);
        let max_ages_buffer = schedule.describe_buffer(&buffer_desc);
        let age_histogram_buffer = schedule.describe_buffer(&BufferDesc::new(MAX_AGE * mem::size_of::<u32>()));

        schedule.add_compute(
            command_name!("generate_image"),
            |params| {
                params.add_image(input_image, ImageUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let generate_image_descriptor_set_layout = &self.generate_image_descriptor_set_layout;
                let generate_image_pipeline_layout = self.generate_image_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let input_image_view = params.get_image_view(input_image);

                    let descriptor_set = generate_image_descriptor_set_layout.write(descriptor_pool, input_image_view);

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        generate_image_pipeline_layout,
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
                params.add_buffer(entries_buffer, BufferUsage::COMPUTE_STORAGE_WRITE);
                params.add_buffer(max_ages_buffer, BufferUsage::COMPUTE_STORAGE_WRITE);
                params.add_buffer(age_histogram_buffer, BufferUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let clear_hash_table_descriptor_set_layout = &self.clear_hash_table_descriptor_set_layout;
                let clear_hash_table_pipeline_layout = self.clear_hash_table_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let max_ages_buffer = params.get_buffer(max_ages_buffer);
                    let age_histogram_buffer = params.get_buffer(age_histogram_buffer);

                    let descriptor_set = clear_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        entries_buffer,
                        max_ages_buffer,
                        age_histogram_buffer,
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        clear_hash_table_pipeline_layout,
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
                params.add_buffer(entries_buffer, BufferUsage::COMPUTE_STORAGE_ATOMIC);
                params.add_buffer(max_ages_buffer, BufferUsage::COMPUTE_STORAGE_ATOMIC);
                params.add_image(input_image, ImageUsage::COMPUTE_STORAGE_READ);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let update_hash_table_descriptor_set_layout = &self.update_hash_table_descriptor_set_layout;
                let update_hash_table_pipeline_layout = self.update_hash_table_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let max_ages_buffer = params.get_buffer(max_ages_buffer);
                    let input_image_view = params.get_image_view(input_image);

                    let descriptor_set = update_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        entries_buffer,
                        max_ages_buffer,
                        input_image_view,
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        update_hash_table_pipeline_layout,
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
                params.add_buffer(entries_buffer, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_buffer(age_histogram_buffer, BufferUsage::COMPUTE_STORAGE_ATOMIC);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let make_age_histogram_descriptor_set_layout = &self.make_age_histogram_descriptor_set_layout;
                let make_age_histogram_pipeline_layout = self.make_age_histogram_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let age_histogram_buffer = params.get_buffer(age_histogram_buffer);

                    let descriptor_set = make_age_histogram_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        entries_buffer,
                        age_histogram_buffer,
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        make_age_histogram_pipeline_layout,
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
                params.add_buffer(entries_buffer, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_buffer(max_ages_buffer, BufferUsage::COMPUTE_STORAGE_READ);
                params.add_image(output_image, ImageUsage::COMPUTE_STORAGE_WRITE);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let update_hash_table_descriptor_set_layout = &self.update_hash_table_descriptor_set_layout;
                let update_hash_table_pipeline_layout = self.update_hash_table_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let max_ages_buffer = params.get_buffer(max_ages_buffer);
                    let output_image_view = params.get_image_view(output_image);

                    let descriptor_set = update_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut HashTableInfo| *buf = hash_table_info,
                        entries_buffer,
                        max_ages_buffer,
                        output_image_view,
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        update_hash_table_pipeline_layout,
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
                params.add_buffer(entries_buffer, BufferUsage::FRAGMENT_STORAGE_READ);
                params.add_buffer(age_histogram_buffer, BufferUsage::FRAGMENT_STORAGE_READ);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let debug_image_descriptor_set_layout = &self.debug_image_descriptor_set_layout;
                let debug_image_pipeline_layout = self.debug_image_pipeline_layout;
                let debug_hash_table_descriptor_set_layout = &self.debug_hash_table_descriptor_set_layout;
                let debug_hash_table_pipeline_layout = self.debug_hash_table_pipeline_layout;
                let debug_age_histogram_descriptor_set_layout = &self.debug_age_histogram_descriptor_set_layout;
                let debug_age_histogram_pipeline_layout = self.debug_age_histogram_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    let input_image_view = params.get_image_view(input_image);
                    let output_image_view = params.get_image_view(output_image);
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let age_histogram_buffer = params.get_buffer(age_histogram_buffer);

                    set_viewport_helper(&context.device, cmd, swap_size);
                    let ortho_from_screen =
                        Scale2Offset2::new(Vec2::broadcast(2.0) / swap_size.as_float(), Vec2::broadcast(-1.0));

                    // visualise results
                    let screen_from_image = Scale2Offset2::new(image_size.as_float(), Vec2::broadcast(10.0));
                    let descriptor_set = debug_image_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_image,
                            };
                        },
                        input_image_view,
                        output_image_view,
                    );
                    let state = GraphicsPipelineState::new(render_pass, main_sample_count)
                        .with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        debug_image_pipeline_layout,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_image.frag.spv",
                        descriptor_set,
                        4,
                    );

                    let screen_from_table = Scale2Offset2::new(Vec2::new(128.0, 1024.0), Vec2::new(1044.0, 10.0));
                    let descriptor_set = debug_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_table,
                            };
                        },
                        |buf: &mut HashTableInfo| {
                            *buf = hash_table_info;
                        },
                        entries_buffer,
                    );
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        debug_hash_table_pipeline_layout,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_hash_table.frag.spv",
                        descriptor_set,
                        4,
                    );

                    let screen_from_histogram = Scale2Offset2::new(Vec2::new(160.0, 80.0), Vec2::new(1184.0, 10.0));
                    let descriptor_set = debug_age_histogram_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut DebugQuadUniforms| {
                            *buf = DebugQuadUniforms {
                                ortho_from_quad: ortho_from_screen * screen_from_histogram,
                            };
                        },
                        |buf: &mut HashTableInfo| {
                            *buf = hash_table_info;
                        },
                        age_histogram_buffer,
                    );
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        debug_age_histogram_pipeline_layout,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_age_histogram.frag.spv",
                        descriptor_set,
                        4,
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
        .with_inner_size(Size::Logical(LogicalSize::new(1920.0, 1080.0)))
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
