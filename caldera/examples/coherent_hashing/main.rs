use caldera::prelude::*;
use imgui::{im_str, Key};
use spark::vk;
use structopt::StructOpt;
use strum::VariantNames;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

descriptor_set_layout!(GenerateImageDescriptorSetLayout { image: StorageImage });

descriptor_set_layout!(DebugImageDescriptorSetLayout { image: StorageImage });

struct App {
    context: SharedContext,

    generate_image_descriptor_set_layout: GenerateImageDescriptorSetLayout,
    generate_image_pipeline_layout: vk::PipelineLayout,
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

        let debug_image_descriptor_set_layout = DebugImageDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_image_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(debug_image_descriptor_set_layout.0);

        Self {
            context,
            generate_image_descriptor_set_layout,
            generate_image_pipeline_layout,
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
