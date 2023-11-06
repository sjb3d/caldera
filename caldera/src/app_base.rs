use crate::prelude::*;
use spark::{vk, Device};
use std::slice;
use winit::{
    event::Event,
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
    shader_name: &str,
    constants: &[SpecializationConstant],
    descriptor_set: DescriptorSet,
    grid_size: UVec2,
) {
    let pipeline_layout = pipeline_cache.get_pipeline_layout(slice::from_ref(&descriptor_set.layout));
    let pipeline = pipeline_cache.get_compute(shader_name, constants, pipeline_layout);
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            slice::from_ref(&descriptor_set.set),
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
    state: &GraphicsPipelineState,
    vertex_shader_name: &str,
    fragment_shader_name: &str,
    descriptor_set: DescriptorSet,
    vertex_count: u32,
) {
    let pipeline_layout = pipeline_cache.get_pipeline_layout(slice::from_ref(&descriptor_set.layout));
    let pipeline = pipeline_cache.get_graphics(
        VertexShaderDesc::standard(vertex_shader_name),
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
            slice::from_ref(&descriptor_set.set),
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
    pub task_system: BackgroundTaskSystem,

    pub descriptor_pool: DescriptorPool,
    pub pipeline_cache: PipelineCache,
    pub command_buffer_pool: CommandBufferPool,
    pub query_pool: QueryPool,

    pub render_graph: RenderGraph,
    pub resource_loader: ResourceLoader,
}

pub struct AppBase {
    pub window: Window,
    pub exit_requested: bool,
    pub context: SharedContext,
    pub display: AppDisplay,
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    pub egui_renderer: spark_egui::Renderer,
    pub systems: AppSystems,
}

impl AppDisplay {
    pub const SWAPCHAIN_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

    pub fn new(context: &SharedContext, window: &Window) -> Self {
        let window_extent = {
            let inner_size = window.inner_size();
            vk::Extent2D {
                width: inner_size.width,
                height: inner_size.height,
            }
        };
        let swapchain = Swapchain::new(context, window_extent, Self::SWAPCHAIN_USAGE);

        Self {
            swapchain,
            recreate_swapchain: false,
        }
    }

    pub fn acquire(&mut self, window: &Window, image_available_semaphore: vk::Semaphore) -> UniqueImage {
        loop {
            if self.recreate_swapchain {
                let window_extent = {
                    let inner_size = window.inner_size();
                    vk::Extent2D {
                        width: inner_size.width,
                        height: inner_size.height,
                    }
                };
                self.swapchain.recreate(window_extent, Self::SWAPCHAIN_USAGE);
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
        let task_system = BackgroundTaskSystem::new();

        let descriptor_pool = DescriptorPool::new(context);
        let pipeline_cache = PipelineCache::new(context, "spv/bin");
        let command_buffer_pool = CommandBufferPool::new(context);
        let query_pool = QueryPool::new(context);

        const CHUNK_SIZE: u32 = 128 * 1024 * 1024;
        const STAGING_SIZE: u32 = 4 * 1024 * 1024;
        let resources = Resources::new(context, CHUNK_SIZE);
        let render_graph = RenderGraph::new(context, &resources, CHUNK_SIZE, CHUNK_SIZE, STAGING_SIZE);
        let resource_loader = ResourceLoader::new(context, &resources, CHUNK_SIZE);

        Self {
            task_system,

            descriptor_pool,
            pipeline_cache,
            command_buffer_pool,
            query_pool,

            render_graph,
            resource_loader,
        }
    }

    pub fn acquire_command_buffer(&mut self) -> CommandBufferAcquireResult {
        let cbar = self.command_buffer_pool.acquire();
        self.pipeline_cache.begin_frame();
        self.render_graph.begin_frame();
        self.descriptor_pool.begin_frame();
        self.query_pool.begin_frame(cbar.pre_swapchain_cmd);
        self.resource_loader.begin_frame(cbar.pre_swapchain_cmd);
        cbar
    }

    pub fn draw_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Memory")
            .default_pos([360.0, 5.0])
            .default_size([270.0, 310.0])
            .default_open(false)
            .show(ctx, |ui| {
                egui::Grid::new("memory_grid").show(ui, |ui| {
                    ui.label("Stat");
                    ui.label("Value");
                    ui.end_row();
                    self.pipeline_cache.ui_stats_table_rows(ui);
                    self.render_graph.ui_stats_table_rows(ui);
                    self.descriptor_pool.ui_stats_table_rows(ui);
                });
            });
        egui::Window::new("Timestamps")
            .default_pos([410.0, 30.0])
            .default_size([220.0, 120.0])
            .show(ctx, |ui| {
                self.query_pool.ui_timestamp_table(ui);
            });
    }

    pub fn submit_command_buffer(&mut self, cbar: &CommandBufferAcquireResult) -> Option<vk::Semaphore> {
        self.query_pool.end_frame(cbar.post_swapchain_cmd);
        self.descriptor_pool.end_frame();
        self.render_graph.end_frame();
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
        let display = AppDisplay::new(&context, &window);

        let egui_max_vertex_count = 64 * 1024;
        let egui_max_texture_side = context
            .physical_device_properties
            .limits
            .max_image_dimension_2d
            .min(2048);

        let egui_ctx = egui::Context::default();
        let mut egui_winit = egui_winit::State::new(&window);
        egui_winit.set_pixels_per_point(window.scale_factor() as f32);
        egui_winit.set_max_texture_side(egui_max_texture_side as usize);
        let egui_renderer = spark_egui::Renderer::new(
            &context.device,
            &context.physical_device_properties,
            &context.physical_device_memory_properties,
            egui_max_vertex_count,
            egui_max_texture_side,
        );

        let systems = AppSystems::new(&context);

        Self {
            window,
            exit_requested: false,
            context,
            display,
            egui_ctx,
            egui_winit,
            egui_renderer,
            systems,
        }
    }

    pub fn ui_begin_frame(&mut self) {
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        self.egui_ctx.begin_frame(raw_input);
    }

    pub fn ui_end_frame(&mut self, cmd: vk::CommandBuffer) {
        let egui::FullOutput {
            platform_output,
            repaint_after: _repaint_after,
            textures_delta,
            shapes,
        } = self.egui_ctx.end_frame();
        self.egui_winit
            .handle_platform_output(&self.window, &self.egui_ctx, platform_output);

        let clipped_primitives = self.egui_ctx.tessellate(shapes);
        self.egui_renderer.update(
            &self.context.device,
            &self.context.physical_device_memory_properties,
            cmd,
            clipped_primitives,
            textures_delta,
        );
    }

    pub fn process_event<T>(
        &mut self,
        event: &Event<'_, T>,
        _target: &EventLoopWindowTarget<T>,
        control_flow: &mut ControlFlow,
    ) -> AppEventResult {
        let mut result = AppEventResult::None;
        match event {
            Event::RedrawEventsCleared => {
                result = AppEventResult::Redraw;
            }
            Event::WindowEvent { event, .. } => {
                let event_response = self.egui_winit.on_event(&self.egui_ctx, &event);
                if event_response.repaint {
                    self.window.request_redraw();
                }
            }
            Event::LoopDestroyed => {
                result = AppEventResult::Destroy;
            }
            _ => {}
        }
        if self.exit_requested {
            control_flow.set_exit();
        } else {
            control_flow.set_poll();
        }
        result
    }
}

impl Drop for AppBase {
    fn drop(&mut self) {
        unsafe { self.context.device.device_wait_idle() }.unwrap();

        self.egui_renderer.destroy(&self.context.device);
    }
}
