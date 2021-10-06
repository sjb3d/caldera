use crate::prelude::*;
use arrayvec::ArrayVec;
use imgui::Ui;
use spark::{vk, Builder};
use std::{ffi::CStr, mem};

/*
    The goal is to manage:

    * Temporary resources (buffers and images, and their views)
    * Render passes and framebuffers
    * Barriers and layout transitions
    * Synchronisation between queues for async compute

    The API should be along the lines of:

    * Register externally provided resources (e.g. swapchain image for this frame)
    * Describe temporary resources
    * Describe commands
      * All commands specify a list of resource views
      * Render commands additionally specify render target views

    Each command is expected to be a set of draw calls or dispatches that do not
    require synchronisation between them.

    Within each command, the caller must manage:

    * Command buffers
    * Pipelines
    * Descriptor sets
    * Vulkan draw and dispatch commands
*/

struct ResourceSet {
    allocator: Allocator,
    buffer_ids: Vec<BufferId>,
    image_ids: Vec<ImageId>,
}

impl ResourceSet {
    fn new(context: &SharedContext, chunk_size: u32) -> Self {
        Self {
            allocator: Allocator::new(context, chunk_size),
            buffer_ids: Vec::new(),
            image_ids: Vec::new(),
        }
    }

    fn begin_frame(&mut self, resources: &mut Resources) {
        self.allocator.reset();
        for id in self.buffer_ids.drain(..) {
            resources.remove_buffer(id);
        }
        for id in self.image_ids.drain(..) {
            resources.remove_image(id);
        }
    }
}

pub struct RenderGraph {
    resources: SharedResources,
    render_cache: RenderCache,
    temp_allocator: Allocator,
    ping_pong_current_set: ResourceSet,
    ping_pong_prev_set: ResourceSet,
    import_set: ResourceSet,
}

impl RenderGraph {
    pub(crate) fn new(
        context: &SharedContext,
        resources: &SharedResources,
        temp_chunk_size: u32,
        ping_pong_chunk_size: u32,
    ) -> Self {
        Self {
            resources: SharedResources::clone(resources),
            render_cache: RenderCache::new(context),
            temp_allocator: Allocator::new(context, temp_chunk_size),
            ping_pong_current_set: ResourceSet::new(context, ping_pong_chunk_size),
            ping_pong_prev_set: ResourceSet::new(context, ping_pong_chunk_size),
            import_set: ResourceSet::new(context, 0),
        }
    }

    pub fn create_buffer(&mut self, desc: &BufferDesc, all_usage: BufferUsage) -> BufferId {
        self.resources
            .lock()
            .unwrap()
            .create_buffer(desc, all_usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    pub fn get_buffer_desc(&self, id: BufferId) -> BufferDesc {
        *self.resources.lock().unwrap().buffer_resource(id).desc()
    }

    pub fn create_image(&mut self, desc: &ImageDesc, all_usage: ImageUsage) -> ImageId {
        self.resources.lock().unwrap().create_image(desc, all_usage)
    }

    pub fn get_image_desc(&self, id: ImageId) -> ImageDesc {
        *self.resources.lock().unwrap().image_resource(id).desc()
    }

    pub fn begin_frame(&mut self) {
        mem::swap(&mut self.ping_pong_current_set, &mut self.ping_pong_prev_set);
        let mut resources = self.resources.lock().unwrap();
        self.ping_pong_current_set.begin_frame(&mut resources);
        self.import_set.begin_frame(&mut resources);
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        self.resources.lock().unwrap().ui_stats_table_rows(ui);

        self.render_cache.ui_stats_table_rows(ui);

        self.temp_allocator.ui_stats_table_rows(ui, "temp memory");

        self.ping_pong_prev_set
            .allocator
            .ui_stats_table_rows(ui, "ping pong memory");

        ui.text("ping pong buffer");
        ui.next_column();
        ui.text(format!("{}", self.ping_pong_prev_set.buffer_ids.len()));
        ui.next_column();

        ui.text("ping pong image");
        ui.next_column();
        ui.text(format!("{}", self.ping_pong_prev_set.image_ids.len()));
        ui.next_column();
    }
}

/// State for a batch of draw calls, becomes a Vulkan sub-pass
#[derive(Clone, Copy)]
pub struct RenderState {
    pub color_output_id: ImageId,
    pub color_clear_value: [f32; 4],
    pub color_temp_id: Option<ImageId>,
    pub depth_temp_id: Option<ImageId>,
}

impl RenderState {
    pub fn new(color_output_id: ImageId, color_clear_value: &[f32; 4]) -> Self {
        Self {
            color_output_id,
            color_clear_value: *color_clear_value,
            color_temp_id: None,
            depth_temp_id: None,
        }
    }

    pub fn with_color_temp(mut self, image_id: ImageId) -> Self {
        self.color_temp_id = Some(image_id);
        self
    }

    pub fn with_depth_temp(mut self, image_id: ImageId) -> Self {
        self.depth_temp_id = Some(image_id);
        self
    }
}

#[derive(Clone, Copy)]
enum ParameterDesc {
    Buffer { id: BufferId, usage: BufferUsage },
    Image { id: ImageId, usage: ImageUsage },
}

struct CommandCommon {
    name: &'static CStr,
    params: RenderParameterDeclaration,
}

enum Command<'graph> {
    Compute {
        common: CommandCommon,
        callback: Box<dyn FnOnce(RenderParameterAccess, vk::CommandBuffer) + 'graph>,
    },
    Graphics {
        common: CommandCommon,
        state: RenderState,
        callback: Box<dyn FnOnce(RenderParameterAccess, vk::CommandBuffer, vk::RenderPass) + 'graph>,
    },
}

struct SplitCommandBuffer {
    current: vk::CommandBuffer,
    next: Option<vk::CommandBuffer>,
    swap_image_id: Option<ImageId>,
}

impl SplitCommandBuffer {
    fn new(first: vk::CommandBuffer, second: vk::CommandBuffer, swap_image_id: Option<ImageId>) -> Self {
        Self {
            current: first,
            next: Some(second),
            swap_image_id,
        }
    }

    fn notify_image_use(&mut self, image_id: ImageId, query_pool: &mut QueryPool) {
        if self.swap_image_id == Some(image_id) {
            query_pool.end_command_buffer(self.current);
            self.current = self.next.take().unwrap();
            self.swap_image_id = None;
        }
    }
}

pub struct RenderParameterDeclaration(Vec<ParameterDesc>);

impl RenderParameterDeclaration {
    fn new() -> Self {
        Self(Vec::new())
    }

    pub fn add_buffer(&mut self, id: BufferId, usage: BufferUsage) {
        self.0.push(ParameterDesc::Buffer { id, usage });
    }

    pub fn add_image(&mut self, id: ImageId, usage: ImageUsage) {
        self.0.push(ParameterDesc::Image { id, usage })
    }
}

pub struct RenderParameterAccess<'graph>(&'graph RenderGraph);

impl<'graph> RenderParameterAccess<'graph> {
    fn new(render_graph: &'graph RenderGraph) -> Self {
        Self(render_graph)
    }

    pub fn get_buffer(&self, id: BufferId) -> vk::Buffer {
        // TODO: cache these, avoid lock per parameter
        self.0.resources.lock().unwrap().buffer_resource(id).buffer().0
    }

    pub fn get_image_view(&self, id: ImageId) -> vk::ImageView {
        // TODO: cache these, avoid lock per parameter
        self.0.resources.lock().unwrap().image_resource(id).image_view().0
    }
}

enum TemporaryId {
    Buffer(BufferId),
    Image(ImageId),
}

enum FinalUsage {
    Buffer { id: BufferId, usage: BufferUsage },
    Image { id: ImageId, usage: ImageUsage },
}

pub struct RenderSchedule<'graph> {
    render_graph: &'graph mut RenderGraph,
    temporaries: Vec<TemporaryId>,
    commands: Vec<Command<'graph>>,
    final_usage: Vec<FinalUsage>,
}

impl<'graph> RenderSchedule<'graph> {
    pub(crate) fn new(render_graph: &'graph mut RenderGraph) -> Self {
        render_graph.temp_allocator.reset();
        RenderSchedule {
            render_graph,
            temporaries: Vec::new(),
            commands: Vec::new(),
            final_usage: Vec::new(),
        }
    }

    pub fn graph(&self) -> &RenderGraph {
        self.render_graph
    }

    pub fn get_bindless_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.render_graph.resources.lock().unwrap().bindless_descriptor_set_layout()
    }

    pub fn get_bindless_descriptor_set(&self) -> vk::DescriptorSet {
        self.render_graph.resources.lock().unwrap().bindless_descriptor_set()
    }

    pub fn get_buffer(&self, id: BufferId) -> vk::Buffer {
        self.render_graph
            .resources
            .lock()
            .unwrap()
            .buffer_resource(id)
            .buffer()
            .0
    }

    pub fn get_image_view(&self, id: ImageId) -> vk::ImageView {
        self.render_graph
            .resources
            .lock()
            .unwrap()
            .image_resource(id)
            .image_view()
            .0
    }

    pub fn create_buffer(&mut self, desc: &BufferDesc, all_usage: BufferUsage) -> BufferId {
        self.render_graph.create_buffer(desc, all_usage)
    }

    pub fn create_image(&mut self, desc: &ImageDesc, all_usage: ImageUsage) -> ImageId {
        self.render_graph.create_image(desc, all_usage)
    }

    pub fn import_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
        final_usage: BufferUsage,
    ) -> BufferId {
        let buffer_id =
            self.render_graph
                .resources
                .lock()
                .unwrap()
                .import_buffer(desc, all_usage, buffer, current_usage);
        self.render_graph.import_set.buffer_ids.push(buffer_id);
        if !final_usage.is_empty() {
            self.final_usage.push(FinalUsage::Buffer {
                id: buffer_id,
                usage: final_usage,
            });
        }
        buffer_id
    }

    pub fn describe_buffer(&mut self, desc: &BufferDesc) -> BufferId {
        let buffer_id = self.render_graph.resources.lock().unwrap().describe_buffer(desc);
        self.temporaries.push(TemporaryId::Buffer(buffer_id));
        buffer_id
    }

    pub fn import_image(
        &mut self,
        desc: &ImageDesc,
        all_usage: ImageUsage,
        image: UniqueImage,
        current_usage: ImageUsage,
        final_usage: ImageUsage,
    ) -> ImageId {
        let image_id = self
            .render_graph
            .resources
            .lock()
            .unwrap()
            .import_image(desc, all_usage, image, current_usage);
        self.render_graph.import_set.image_ids.push(image_id);
        if !final_usage.is_empty() {
            self.final_usage.push(FinalUsage::Image {
                id: image_id,
                usage: final_usage,
            });
        }
        image_id
    }

    pub fn describe_image(&mut self, desc: &ImageDesc) -> ImageId {
        let image_id = self.render_graph.resources.lock().unwrap().describe_image(desc);
        self.temporaries.push(TemporaryId::Image(image_id));
        image_id
    }

    pub fn add_compute(
        &mut self,
        name: &'static CStr,
        decl: impl FnOnce(&mut RenderParameterDeclaration),
        callback: impl FnOnce(RenderParameterAccess, vk::CommandBuffer) + 'graph,
    ) {
        let mut params = RenderParameterDeclaration::new();
        decl(&mut params);
        self.commands.push(Command::Compute {
            common: CommandCommon { name, params },
            callback: Box::new(callback),
        });
    }

    pub fn add_graphics(
        &mut self,
        name: &'static CStr,
        state: RenderState,
        decl: impl FnOnce(&mut RenderParameterDeclaration),
        callback: impl FnOnce(RenderParameterAccess, vk::CommandBuffer, vk::RenderPass) + 'graph,
    ) {
        let mut params = RenderParameterDeclaration::new();
        decl(&mut params);
        self.commands.push(Command::Graphics {
            common: CommandCommon { name, params },
            state,
            callback: Box::new(callback),
        });
    }

    pub fn run(
        mut self,
        context: &Context,
        pre_swapchain_cmd: vk::CommandBuffer,
        post_swapchain_cmd: vk::CommandBuffer,
        swap_image_id: Option<ImageId>,
        query_pool: &mut QueryPool,
    ) {
        // loop over commands to set usage for all resources
        {
            let mut resources = self.render_graph.resources.lock().unwrap();
            for command in &self.commands {
                let common = match command {
                    Command::Compute { common, .. } => common,
                    Command::Graphics { common, state, .. } => {
                        resources
                            .image_resource_mut(state.color_output_id)
                            .declare_usage(ImageUsage::COLOR_ATTACHMENT_WRITE);
                        if let Some(id) = state.color_temp_id {
                            resources
                                .image_resource_mut(id)
                                .declare_usage(ImageUsage::TRANSIENT_COLOR_ATTACHMENT);
                        }
                        if let Some(id) = state.depth_temp_id {
                            resources
                                .image_resource_mut(id)
                                .declare_usage(ImageUsage::TRANSIENT_DEPTH_ATTACHMENT)
                        }
                        common
                    }
                };
                for parameter_desc in &common.params.0 {
                    match parameter_desc {
                        ParameterDesc::Buffer { id, usage } => {
                            resources.buffer_resource_mut(*id).declare_usage(*usage);
                        }
                        ParameterDesc::Image { id, usage } => {
                            resources.image_resource_mut(*id).declare_usage(*usage);
                        }
                    }
                }
            }

            // allocate temporaries
            let temp_allocator = &mut self.render_graph.temp_allocator;
            for id in &self.temporaries {
                match id {
                    TemporaryId::Buffer(id) => resources.allocate_temporary_buffer(*id, temp_allocator),
                    TemporaryId::Image(id) => resources.allocate_temporary_image(*id, temp_allocator),
                }
            }
        }

        // for now we just emit single barriers just in time
        /*
            TODO: build graph of commands, barriers, and layout changes

            Goals would be to:
            * Do barriers and layout changes early
            * Combine usage where possible (e.g. read by different stages)
            * Combine with render pass for attachments where possible

            TODO: think about how to do sub-passes... probably explicit is better?
        */
        let mut cmd = SplitCommandBuffer::new(pre_swapchain_cmd, post_swapchain_cmd, swap_image_id);
        for command in self.commands.drain(..) {
            let common = match &command {
                Command::Compute { common, .. } => common,
                Command::Graphics { common, .. } => common,
            };

            let is_debug = context.instance.extensions.ext_debug_utils;
            if is_debug {
                let label = vk::DebugUtilsLabelEXT {
                    p_label_name: common.name.as_ptr(),
                    ..Default::default()
                };
                unsafe { context.instance.cmd_begin_debug_utils_label_ext(cmd.current, &label) };
            }

            {
                let mut resources = self.render_graph.resources.lock().unwrap();
                for parameter_desc in &common.params.0 {
                    match parameter_desc {
                        ParameterDesc::Buffer { id, usage } => {
                            resources.transition_buffer_usage(*id, *usage, context, cmd.current);
                        }
                        ParameterDesc::Image { id, usage } => {
                            cmd.notify_image_use(*id, query_pool);
                            resources.transition_image_usage(*id, *usage, context, cmd.current);
                        }
                    }
                }
            }

            let timestamp_name = common.name;
            match command {
                Command::Compute { callback, .. } => {
                    query_pool.emit_timestamp(cmd.current, timestamp_name);
                    (callback)(RenderParameterAccess::new(self.render_graph), cmd.current);
                }
                Command::Graphics { state, callback, .. } => {
                    let render_pass = {
                        let resources = self.render_graph.resources.lock().unwrap();

                        // TODO: initial layout as part of render pass (assumes UNDEFINED for now)
                        cmd.notify_image_use(state.color_output_id, query_pool);

                        let mut clear_values = ArrayVec::<_, { RenderCache::MAX_ATTACHMENTS }>::new();
                        let color_clear_value = vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: state.color_clear_value,
                            },
                        };
                        let depth_clear_value = vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                        };

                        // color output (always present for now)
                        clear_values.push(color_clear_value);
                        let color_output_resource = resources.image_resource(state.color_output_id);
                        let color_output_desc = color_output_resource.desc();
                        let color_output_image_view = color_output_resource.image_view();

                        // color temp (if present)
                        let (samples, color_temp_image_view) = if let Some(color_temp_id) = state.color_temp_id {
                            let color_temp_resource = resources.image_resource(color_temp_id);
                            let color_temp_desc = color_temp_resource.desc();
                            let color_temp_image_view = color_temp_resource.image_view();

                            assert_eq!(color_output_desc.size(), color_temp_desc.size());
                            assert_eq!(color_output_desc.format, color_temp_desc.format);

                            clear_values.push(color_clear_value);

                            (color_temp_desc.samples, Some(color_temp_image_view))
                        } else {
                            (vk::SampleCountFlags::N1, None)
                        };

                        // depth temp (if present)
                        let (depth_format, depth_temp_image_view) = if let Some(depth_temp_id) = state.depth_temp_id {
                            let depth_temp_resource = resources.image_resource(depth_temp_id);
                            let depth_temp_desc = depth_temp_resource.desc();
                            let depth_temp_image_view = depth_temp_resource.image_view();

                            assert_eq!(color_output_desc.size(), depth_temp_desc.size());
                            assert_eq!(samples, depth_temp_desc.samples);

                            clear_values.push(depth_clear_value);

                            (Some(depth_temp_desc.format), Some(depth_temp_image_view))
                        } else {
                            (None, None)
                        };

                        let render_pass = self.render_graph.render_cache.get_render_pass(
                            color_output_desc.format,
                            depth_format,
                            samples,
                        );

                        let framebuffer = self.render_graph.render_cache.get_framebuffer(
                            render_pass,
                            color_output_desc.size(),
                            color_output_image_view,
                            color_temp_image_view,
                            depth_temp_image_view,
                        );

                        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                            .render_pass(render_pass.0)
                            .framebuffer(framebuffer.0)
                            .render_area(vk::Rect2D {
                                offset: Default::default(),
                                extent: color_output_desc.extent_2d(),
                            })
                            .p_clear_values(&clear_values);
                        unsafe {
                            context.device.cmd_begin_render_pass(
                                cmd.current,
                                &render_pass_begin_info,
                                vk::SubpassContents::INLINE,
                            )
                        };

                        render_pass
                    };

                    query_pool.emit_timestamp(cmd.current, timestamp_name);

                    (callback)(
                        RenderParameterAccess::new(self.render_graph),
                        cmd.current,
                        render_pass.0,
                    );

                    unsafe { context.device.cmd_end_render_pass(cmd.current) };

                    // TODO: final layout as part of render pass
                    self.render_graph
                        .resources
                        .lock()
                        .unwrap()
                        .image_resource_mut(state.color_output_id)
                        .force_usage(ImageUsage::COLOR_ATTACHMENT_WRITE);
                }
            }

            if is_debug {
                unsafe { context.instance.cmd_end_debug_utils_label_ext(cmd.current) };
            }
        }

        {
            let mut resources = self.render_graph.resources.lock().unwrap();

            // free temporaries
            for id in self.temporaries.drain(..) {
                match id {
                    TemporaryId::Buffer(id) => resources.remove_buffer(id),
                    TemporaryId::Image(id) => resources.remove_image(id),
                }
            }

            // emit any last barriers (usually at least the swap chain image)
            for final_usage in self.final_usage.drain(..) {
                match final_usage {
                    FinalUsage::Buffer { id, usage } => {
                        resources.transition_buffer_usage(id, usage, context, cmd.current);
                    }
                    FinalUsage::Image { id, usage } => {
                        cmd.notify_image_use(id, query_pool);
                        resources.transition_image_usage(id, usage, context, cmd.current);
                    }
                }
            }
        }
    }
}
