use caldera::*;
use imgui::im_str;
use imgui::Key;
use spark::vk;
use std::env;
use std::sync::Arc;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

struct App {
    context: Arc<Context>,
}

impl App {
    fn new(base: &mut AppBase) -> Self {
        let context = &base.context;

        Self {
            context: Arc::clone(&context),
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
            .build(&ui, {
                let ui = &ui;
                move || {
                    ui.text("Hello, World!");
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

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
        let main_render_state = RenderState::new(swap_image, &[0f32, 0f32, 0.1f32, 0f32]);

        schedule.add_graphics(command_name!("main"), main_render_state, |_params| {}, {
            let context = &base.context;
            let pipeline_cache = &base.systems.pipeline_cache;
            let window = &base.window;
            let ui_platform = &mut base.ui_platform;
            let ui_renderer = &mut base.ui_renderer;
            move |_params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_extent);

                // draw imgui
                ui_platform.prepare_render(&ui, window);

                let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
            }
        });

        schedule.run(
            &base.context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            swap_image,
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display.present(swap_vk_image, rendering_finished_semaphore);
    }
}

fn main() {
    let mut params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        allow_ray_tracing: true,
        ..Default::default()
    };
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        if !params.parse_arg(arg) {
            panic!("unknown argument {:?}", arg);
        }
    }

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("mesh")
        .with_inner_size(Size::Logical(LogicalSize::new(640.0, 480.0)))
        .build(&event_loop)
        .unwrap();

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
