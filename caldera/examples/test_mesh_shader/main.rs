use caldera::prelude::*;
use imgui::Key;
use spark::vk;
use std::sync::Arc;
use structopt::StructOpt;
use strum::VariantNames;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

struct App {
    context: SharedContext,
}

impl App {
    fn new(base: &mut AppBase) -> Self {
        let context = Arc::clone(&base.context);

        Self { context }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(Key::Escape) {
            base.exit_requested = true;
        }

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
        let main_render_state = RenderState::new(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]);

        schedule.add_graphics(command_name!("main"), main_render_state, |_params| {}, {
            let context = &base.context;
            let pipeline_cache = &base.systems.pipeline_cache;
            let window = &base.window;
            let ui_platform = &mut base.ui_platform;
            let ui_renderer = &mut base.ui_renderer;
            move |_params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_size);

                // draw imgui
                ui_platform.prepare_render(&ui, window);

                let pipeline = pipeline_cache.get_ui(ui_renderer, render_pass, vk::SampleCountFlags::N1);
                ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
            }
        });

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
    #[structopt(short, long, parse(try_from_str=try_version_from_str), default_value="1.0")]
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
        .with_title("test_mesh_shader")
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