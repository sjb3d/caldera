use crate::prelude::*;
use imgui::im_str;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use spark::{vk, Device};
use std::{slice, time::Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopWindowTarget},
    window::Window,
};

pub fn set_viewport_helper(device: &Device, cmd: vk::CommandBuffer, size: UVec2) {
    let viewport = vk::Viewport {
        width: size.x as f32,
        height: size.y as f32,
        max_depth: 1.0,
        ..Default::default()
    };
    let scissor = vk::Rect2D {
        extent: vk::Extent2D {
            width: size.x,
            height: size.y,
        },
        ..Default::default()
    };
    unsafe {
        device.cmd_set_viewport(cmd, 0, slice::from_ref(&viewport));
        device.cmd_set_scissor(cmd, 0, slice::from_ref(&scissor));
    }
}

pub fn dispatch_helper(
    device: &Device,
    pipeline_cache: &PipelineCache,
    cmd: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    shader_name: &str,
    constants: &[SpecializationConstant],
    descriptor_set: vk::DescriptorSet,
    grid_size: UVec2,
) {
    let pipeline = pipeline_cache.get_compute(shader_name, constants, pipeline_layout);
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            slice::from_ref(&descriptor_set),
            &[],
        );
        device.cmd_dispatch(cmd, grid_size.x, grid_size.y, 1);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_helper(
    device: &Device,
    pipeline_cache: &PipelineCache,
    cmd: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    state: &GraphicsPipelineState,
    vertex_shader_name: &str,
    fragment_shader_name: &str,
    descriptor_set: vk::DescriptorSet,
    vertex_count: u32,
) {
    let pipeline = pipeline_cache.get_graphics(
        VertexShaderNames::standard(vertex_shader_name),
        fragment_shader_name,
        pipeline_layout,
        state,
    );
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            slice::from_ref(&descriptor_set),
            &[],
        );
        device.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }
}

pub struct AppDisplay {
    pub swapchain: Swapchain,
    pub recreate_swapchain: bool,
}

pub struct AppSystems {
    pub descriptor_set_layout_cache: DescriptorSetLayoutCache,
    pub descriptor_pool: DescriptorPool,
    pub pipeline_cache: PipelineCache,
    pub command_buffer_pool: CommandBufferPool,
    pub query_pool: QueryPool,

    pub resource_loader: ResourceLoader,
    pub global_allocator: Allocator, // TODO: fix interactions with resource loader (needs to drop after)
    pub render_graph: RenderGraph,
}

pub struct AppBase {
    pub window: Window,
    pub exit_requested: bool,
    pub context: SharedContext,
    pub display: AppDisplay,
    pub last_instant: Instant,
    pub ui_context: imgui::Context,
    pub ui_platform: WinitPlatform,
    pub ui_renderer: spark_imgui::Renderer,
    pub systems: AppSystems,
}

impl AppDisplay {
    pub const SWAPCHAIN_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

    pub fn new(context: &SharedContext) -> Self {
        let swapchain = Swapchain::new(context, Self::SWAPCHAIN_USAGE);

        Self {
            swapchain,
            recreate_swapchain: false,
        }
    }

    pub fn acquire(&mut self, image_available_semaphore: vk::Semaphore) -> UniqueImage {
        loop {
            if self.recreate_swapchain {
                self.swapchain.recreate(Self::SWAPCHAIN_USAGE);
                self.recreate_swapchain = false;
            }
            match self.swapchain.acquire(image_available_semaphore) {
                SwapchainAcquireResult::Ok(image) => break image,
                SwapchainAcquireResult::RecreateSoon(image) => {
                    self.recreate_swapchain = true;
                    break image;
                }
                SwapchainAcquireResult::RecreateNow => self.recreate_swapchain = true,
            };
        }
    }

    pub fn present(&mut self, swap_image: UniqueImage, rendering_finished_semaphore: vk::Semaphore) {
        self.swapchain.present(swap_image, rendering_finished_semaphore);
    }
}

impl AppSystems {
    pub fn new(context: &SharedContext) -> Self {
        let descriptor_set_layout_cache = DescriptorSetLayoutCache::new(context);
        let descriptor_pool = DescriptorPool::new(context);
        let pipeline_cache = PipelineCache::new(context, "spv/bin");
        let command_buffer_pool = CommandBufferPool::new(context);
        let query_pool = QueryPool::new(context);

        const CHUNK_SIZE: u32 = 128 * 1024 * 1024;
        let mut global_allocator = Allocator::new(context, CHUNK_SIZE);
        let resource_loader = ResourceLoader::new(context, &mut global_allocator, CHUNK_SIZE);
        let render_graph = RenderGraph::new(context, CHUNK_SIZE, CHUNK_SIZE);

        Self {
            descriptor_set_layout_cache,
            descriptor_pool,
            pipeline_cache,
            command_buffer_pool,
            query_pool,

            global_allocator,
            resource_loader,
            render_graph,
        }
    }

    pub fn acquire_command_buffer(&mut self) -> CommandBufferAcquireResult {
        let cbar = self.command_buffer_pool.acquire();
        self.pipeline_cache.begin_frame();
        self.render_graph.begin_frame();
        self.descriptor_pool.begin_frame();
        self.query_pool.begin_frame(cbar.pre_swapchain_cmd);
        self.resource_loader
            .begin_frame(&mut self.global_allocator, cbar.pre_swapchain_cmd);
        cbar
    }

    pub fn draw_ui(&mut self, ui: &imgui::Ui) {
        imgui::Window::new(im_str!("Memory"))
            .position([360.0, 5.0], imgui::Condition::FirstUseEver)
            .size([270.0, 310.0], imgui::Condition::FirstUseEver)
            .collapsed(true, imgui::Condition::Once)
            .build(ui, || {
                ui.columns(2, im_str!("StatsBegin"), true);
                ui.text("Stat");
                ui.next_column();
                ui.text("Value");
                ui.next_column();
                ui.separator();
                self.pipeline_cache.ui_stats_table_rows(ui);
                self.render_graph.ui_stats_table_rows(ui);
                self.descriptor_pool.ui_stats_table_rows(ui);
                self.global_allocator.ui_stats_table_rows(ui, "global memory");
                self.resource_loader.ui_stats_table_rows(ui);
                ui.columns(1, im_str!("StatsEnd"), false);
            });
        imgui::Window::new(im_str!("Timestamps"))
            .position([410.0, 30.0], imgui::Condition::FirstUseEver)
            .size([220.0, 120.0], imgui::Condition::FirstUseEver)
            .build(ui, || {
                self.query_pool.ui_timestamp_table(ui);
            });
    }

    pub fn submit_command_buffer(&mut self, cbar: &CommandBufferAcquireResult) -> Option<vk::Semaphore> {
        self.query_pool.end_command_buffer(cbar.post_swapchain_cmd);
        self.descriptor_pool.end_command_buffer();
        self.command_buffer_pool.submit()
    }
}

pub enum AppEventResult {
    None,
    Redraw,
    Destroy,
}

impl AppBase {
    pub fn new(window: Window, params: &ContextParams) -> Self {
        let context = Context::new(Some(&window), params);
        let display = AppDisplay::new(&context);

        let mut ui_context = imgui::Context::create();
        ui_context.fonts().add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: 13.0,
                ..Default::default()
            }),
        }]);

        let mut ui_platform = WinitPlatform::init(&mut ui_context);
        ui_platform.attach_window(ui_context.io_mut(), &window, HiDpiMode::Default);

        let ui_renderer = spark_imgui::Renderer::new(
            &context.device,
            &context.physical_device_properties,
            &context.physical_device_memory_properties,
            &mut ui_context,
        );
        let systems = AppSystems::new(&context);

        Self {
            window,
            exit_requested: false,
            context,
            display,
            last_instant: Instant::now(),
            ui_context,
            ui_platform,
            ui_renderer,
            systems,
        }
    }

    pub fn process_event<T>(
        &mut self,
        event: &Event<'_, T>,
        _target: &EventLoopWindowTarget<T>,
        control_flow: &mut ControlFlow,
    ) -> AppEventResult {
        let mut result = AppEventResult::None;
        match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                self.ui_context.io_mut().update_delta_time(now - self.last_instant);
                self.last_instant = now;
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                self.exit_requested = true;
            }
            Event::MainEventsCleared => {
                self.ui_platform
                    .prepare_frame(self.ui_context.io_mut(), &self.window)
                    .expect("failed to prepare frame");
                self.window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                result = AppEventResult::Redraw;
            }
            Event::LoopDestroyed => {
                result = AppEventResult::Destroy;
            }
            event => {
                self.ui_platform
                    .handle_event(self.ui_context.io_mut(), &self.window, event);
            }
        }

        *control_flow = if self.exit_requested {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };

        result
    }
}

impl Drop for AppBase {
    fn drop(&mut self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();

        self.ui_renderer.delete(&self.context.device);
    }
}
