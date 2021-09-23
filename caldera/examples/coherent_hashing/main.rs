use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use imgui::{im_str, Key};
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

descriptor_set_layout!(GenerateImageDescriptorSetLayout { image: StorageImage });

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearHashTableUniforms {
    entry_count: u32,
}

descriptor_set_layout!(ClearHashTableDescriptorSetLayout {
    uniforms: UniformData<ClearHashTableUniforms>,
    table: StorageBuffer,
});

const MAX_AGE: usize = 15;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WriteHashTableUniforms {
    entry_count: u32,
    offsets: [u32; MAX_AGE],
}

descriptor_set_layout!(WriteHashTableDescriptorSetLayout {
    uniforms: UniformData<WriteHashTableUniforms>,
    table: StorageBuffer,
    image: StorageImage,
});

descriptor_set_layout!(DebugImageDescriptorSetLayout { image: StorageImage });

struct App {
    context: SharedContext,

    generate_image_descriptor_set_layout: GenerateImageDescriptorSetLayout,
    generate_image_pipeline_layout: vk::PipelineLayout,
    clear_hash_table_descriptor_set_layout: ClearHashTableDescriptorSetLayout,
    clear_hash_table_pipeline_layout: vk::PipelineLayout,
    write_hash_table_descriptor_set_layout: WriteHashTableDescriptorSetLayout,
    write_hash_table_pipeline_layout: vk::PipelineLayout,
    debug_image_descriptor_set_layout: DebugImageDescriptorSetLayout,
    debug_image_pipeline_layout: vk::PipelineLayout,

    counter: u32,
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

        let write_hash_table_descriptor_set_layout =
            WriteHashTableDescriptorSetLayout::new(descriptor_set_layout_cache);
        let write_hash_table_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(write_hash_table_descriptor_set_layout.0);

        let debug_image_descriptor_set_layout = DebugImageDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_image_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(debug_image_descriptor_set_layout.0);

        Self {
            context,
            generate_image_descriptor_set_layout,
            generate_image_pipeline_layout,
            clear_hash_table_descriptor_set_layout,
            clear_hash_table_pipeline_layout,
            write_hash_table_descriptor_set_layout,
            write_hash_table_pipeline_layout,
            debug_image_descriptor_set_layout,
            debug_image_pipeline_layout,
            counter: 0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(Key::Escape) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                let ui = &ui;
                let counter = self.counter;
                move || {
                    ui.text(format!("Counter: {}", counter));
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

        let mut primes = Primes::new();
        let entry_count = primes.next_after(10_000);
        let offsets: [u32; MAX_AGE] = {
            let mut rng = SmallRng::seed_from_u64(0);
            let dist = Uniform::new(1_000_000, 10_000_000);
            let mut offsets = ArrayVec::new();
            for _ in 0..MAX_AGE {
                offsets.push(rng.sample(dist));
            }
            offsets.into_inner().unwrap()
        };

        let entries_desc = BufferDesc::new((entry_count as usize) * mem::size_of::<u32>());
        let entries_buffer = schedule.describe_buffer(&entries_desc);

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
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let clear_hash_table_descriptor_set_layout = &self.clear_hash_table_descriptor_set_layout;
                let clear_hash_table_pipeline_layout = self.clear_hash_table_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);

                    let descriptor_set = clear_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut ClearHashTableUniforms| *buf = ClearHashTableUniforms { entry_count },
                        entries_buffer,
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
                params.add_image(input_image, ImageUsage::COMPUTE_STORAGE_READ);
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let write_hash_table_descriptor_set_layout = &self.write_hash_table_descriptor_set_layout;
                let write_hash_table_pipeline_layout = self.write_hash_table_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                move |params, cmd| {
                    let entries_buffer = params.get_buffer(entries_buffer);
                    let input_image_view = params.get_image_view(input_image);

                    let descriptor_set = write_hash_table_descriptor_set_layout.write(
                        descriptor_pool,
                        |buf: &mut WriteHashTableUniforms| *buf = WriteHashTableUniforms { entry_count, offsets },
                        entries_buffer,
                        input_image_view,
                    );

                    dispatch_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        write_hash_table_pipeline_layout,
                        "coherent_hashing/write_hash_table.comp.spv",
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
            },
            {
                let context = base.context.as_ref();
                let descriptor_pool = &base.systems.descriptor_pool;
                let debug_image_descriptor_set_layout = &self.debug_image_descriptor_set_layout;
                let debug_image_pipeline_layout = self.debug_image_pipeline_layout;
                let pipeline_cache = &base.systems.pipeline_cache;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    let image_view = params.get_image_view(input_image);

                    set_viewport_helper(&context.device, cmd, swap_size);

                    // visualise results
                    let descriptor_set = debug_image_descriptor_set_layout.write(descriptor_pool, image_view);
                    let state = GraphicsPipelineState::new(render_pass, main_sample_count);
                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        debug_image_pipeline_layout,
                        &state,
                        "coherent_hashing/debug_quad.vert.spv",
                        "coherent_hashing/debug_image.frag.spv",
                        descriptor_set,
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

        self.counter += 1;
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
