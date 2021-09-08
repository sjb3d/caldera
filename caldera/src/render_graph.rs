use crate::{prelude::*, resource::*};
use arrayvec::ArrayVec;
use imgui::Ui;
use spark::{vk, Builder, Device};
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BufferHandle(ResourceHandle);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ImageHandle(ResourceHandle);

enum BufferResource {
    Temporary {
        desc: BufferDesc,
        all_usage: BufferUsage,
    },
    Ready {
        desc: BufferDesc,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
        all_usage_check: BufferUsage,
    },
}

impl BufferResource {
    fn desc(&self) -> &BufferDesc {
        match self {
            BufferResource::Temporary { desc, .. } => desc,
            BufferResource::Ready { desc, .. } => desc,
        }
    }

    fn buffer(&self) -> Option<UniqueBuffer> {
        match self {
            BufferResource::Ready { buffer, .. } => Some(*buffer),
            _ => None,
        }
    }

    fn declare_usage(&mut self, usage: BufferUsage) {
        match self {
            BufferResource::Temporary { ref mut all_usage, .. } => {
                *all_usage |= usage;
            }
            BufferResource::Ready { all_usage_check, .. } => {
                if !all_usage_check.contains(usage) {
                    panic!("buffer usage {:?} was not declared in {:?}", usage, all_usage_check);
                }
            }
        }
    }

    fn transition_usage(&mut self, new_usage: BufferUsage, device: &Device, cmd: vk::CommandBuffer) {
        match self {
            BufferResource::Ready {
                buffer,
                ref mut current_usage,
                all_usage_check,
                ..
            } => {
                if !all_usage_check.contains(new_usage) {
                    panic!("cannot set usage that buffer was not allocated with");
                }
                if *current_usage != new_usage || !new_usage.as_access_category().supports_overlap() {
                    emit_buffer_barrier(*current_usage, new_usage, buffer.0, device, cmd);
                    *current_usage = new_usage;
                }
            }
            _ => panic!("image not ready"),
        }
    }
}

enum ImageResource {
    Temporary {
        desc: ImageDesc,
        all_usage: ImageUsage,
    },
    Ready {
        desc: ImageDesc,
        image: UniqueImage,
        image_view: UniqueImageView,
        current_usage: ImageUsage,
        all_usage_check: ImageUsage,
    },
}

impl ImageResource {
    fn desc(&self) -> &ImageDesc {
        match self {
            ImageResource::Temporary { desc, .. } => desc,
            ImageResource::Ready { desc, .. } => desc,
        }
    }

    fn image_view(&self) -> Option<UniqueImageView> {
        match self {
            ImageResource::Ready { image_view, .. } => Some(*image_view),
            _ => None,
        }
    }

    fn declare_usage(&mut self, usage: ImageUsage) {
        match self {
            ImageResource::Temporary { ref mut all_usage, .. } => {
                *all_usage |= usage;
            }
            ImageResource::Ready { all_usage_check, .. } => {
                if !all_usage_check.contains(usage) {
                    panic!("image usage {:?} was not declared in {:?}", usage, all_usage_check);
                }
            }
        }
    }

    fn transition_usage(&mut self, new_usage: ImageUsage, device: &Device, cmd: vk::CommandBuffer) {
        match self {
            ImageResource::Ready {
                desc,
                image,
                ref mut current_usage,
                all_usage_check,
                ..
            } => {
                if !all_usage_check.contains(new_usage) {
                    panic!("cannot set usage that image was not allocated with");
                }
                if *current_usage != new_usage || !new_usage.as_access_category().supports_overlap() {
                    emit_image_barrier(*current_usage, new_usage, image.0, desc.aspect_mask, device, cmd);
                    *current_usage = new_usage;
                }
            }
            _ => panic!("image not ready"),
        }
    }

    fn force_usage(&mut self, new_usage: ImageUsage) {
        match self {
            ImageResource::Ready {
                all_usage_check,
                ref mut current_usage,
                ..
            } => {
                if !all_usage_check.contains(new_usage) {
                    panic!("cannot set usage that image was not allocated with");
                }
                *current_usage = new_usage;
            }
            _ => panic!("image not ready"),
        }
    }
}

struct ResourceSet {
    allocator: Allocator,
    buffers: Vec<BufferHandle>,
    images: Vec<ImageHandle>,
}

impl ResourceSet {
    fn new(context: &SharedContext, chunk_size: u32) -> Self {
        Self {
            allocator: Allocator::new(context, chunk_size),
            buffers: Vec::new(),
            images: Vec::new(),
        }
    }

    fn begin_frame(&mut self, buffers: &mut ResourceVec<BufferResource>, images: &mut ResourceVec<ImageResource>) {
        self.allocator.reset();
        for buffer in self.buffers.drain(..) {
            buffers.free(buffer.0);
        }
        for image in self.images.drain(..) {
            images.free(image.0);
        }
    }
}

pub struct RenderGraph {
    resource_cache: ResourceCache,
    render_cache: RenderCache,
    buffers: ResourceVec<BufferResource>,
    images: ResourceVec<ImageResource>,
    temp_allocator: Allocator,
    ping_pong_current_set: ResourceSet,
    ping_pong_prev_set: ResourceSet,
    import_set: ResourceSet,
}

impl RenderGraph {
    pub fn new(context: &SharedContext, temp_chunk_size: u32, ping_pong_chunk_size: u32) -> Self {
        Self {
            resource_cache: ResourceCache::new(context),
            render_cache: RenderCache::new(context),
            buffers: ResourceVec::new(),
            images: ResourceVec::new(),
            temp_allocator: Allocator::new(context, temp_chunk_size),
            ping_pong_current_set: ResourceSet::new(context, ping_pong_chunk_size),
            ping_pong_prev_set: ResourceSet::new(context, ping_pong_chunk_size),
            import_set: ResourceSet::new(context, 0),
        }
    }

    fn create_ready_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
    ) -> BufferHandle {
        BufferHandle(self.buffers.allocate(BufferResource::Ready {
            desc: *desc,
            buffer,
            current_usage,
            all_usage_check: all_usage,
        }))
    }

    pub fn create_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        allocator: &mut Allocator,
    ) -> BufferHandle {
        let buffer = {
            let all_usage_flags = all_usage.as_flags();
            let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
            let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
            self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags)
        };
        self.create_ready_buffer(desc, all_usage, buffer, BufferUsage::empty())
    }

    fn allocate_temporary_buffer(&mut self, handle: BufferHandle) {
        let buffer_resource = self.buffers.get_mut(handle.0).unwrap();
        *buffer_resource = match buffer_resource {
            BufferResource::Temporary { desc, all_usage } => {
                let all_usage_flags = all_usage.as_flags();
                let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
                let alloc = self.temp_allocator.allocate(&info.mem_req, memory_property_flags);
                let buffer = self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags);
                BufferResource::Ready {
                    desc: *desc,
                    buffer,
                    current_usage: BufferUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("buffer is not temporary"),
        };
    }

    pub fn get_buffer_desc(&self, handle: BufferHandle) -> &BufferDesc {
        self.buffers.get(handle.0).unwrap().desc()
    }

    fn free_buffer(&mut self, handle: BufferHandle) {
        self.buffers.free(handle.0);
    }

    fn create_ready_image(
        &mut self,
        desc: &ImageDesc,
        all_usage: ImageUsage,
        image: UniqueImage,
        current_usage: ImageUsage,
    ) -> ImageHandle {
        let image_view = self.resource_cache.get_image_view(desc, image);
        ImageHandle(self.images.allocate(ImageResource::Ready {
            desc: *desc,
            image,
            image_view,
            current_usage,
            all_usage_check: all_usage,
        }))
    }

    pub fn create_image(&mut self, desc: &ImageDesc, all_usage: ImageUsage, allocator: &mut Allocator) -> ImageHandle {
        let image = {
            let all_usage_flags = all_usage.as_flags();
            let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let info = self.resource_cache.get_image_info(desc, all_usage_flags);
            let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
            self.resource_cache.get_image(desc, &info, &alloc, all_usage_flags)
        };
        self.create_ready_image(desc, all_usage, image, ImageUsage::empty())
    }

    fn allocate_temporary_image(&mut self, handle: ImageHandle) {
        let image_resource = self.images.get_mut(handle.0).unwrap();
        *image_resource = match image_resource {
            ImageResource::Temporary { desc, all_usage } => {
                let all_usage_flags = all_usage.as_flags();
                let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                let info = self.resource_cache.get_image_info(desc, all_usage_flags);
                let alloc = self.temp_allocator.allocate(&info.mem_req, memory_property_flags);
                let image = self.resource_cache.get_image(desc, &info, &alloc, all_usage_flags);
                let image_view = self.resource_cache.get_image_view(desc, image);
                ImageResource::Ready {
                    desc: *desc,
                    image,
                    image_view,
                    current_usage: ImageUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("image is not temporary"),
        };
    }

    pub fn get_image_desc(&self, handle: ImageHandle) -> &ImageDesc {
        self.images.get(handle.0).unwrap().desc()
    }

    fn free_image(&mut self, handle: ImageHandle) {
        self.images.free(handle.0);
    }

    pub fn begin_frame(&mut self) {
        mem::swap(&mut self.ping_pong_current_set, &mut self.ping_pong_prev_set);
        self.ping_pong_current_set
            .begin_frame(&mut self.buffers, &mut self.images);

        self.import_set.begin_frame(&mut self.buffers, &mut self.images);
    }

    fn transition_buffer_usage(
        &mut self,
        handle: BufferHandle,
        new_usage: BufferUsage,
        context: &Context,
        cmd: vk::CommandBuffer,
    ) {
        self.buffers
            .get_mut(handle.0)
            .unwrap()
            .transition_usage(new_usage, &context.device, cmd);
    }

    fn transition_image_usage(
        &mut self,
        handle: ImageHandle,
        new_usage: ImageUsage,
        context: &Context,
        cmd: &mut SplitCommandBuffer,
        query_pool: &mut QueryPool,
    ) {
        cmd.notify_image_use(handle, query_pool);
        self.images
            .get_mut(handle.0)
            .unwrap()
            .transition_usage(new_usage, &context.device, cmd.current);
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        ui.text("graph buffers");
        ui.next_column();
        ui.text(format!("{}", self.buffers.active_count()));
        ui.next_column();

        ui.text("graph images");
        ui.next_column();
        ui.text(format!("{}", self.images.active_count()));
        ui.next_column();

        self.resource_cache.ui_stats_table_rows(ui, "graph");
        self.render_cache.ui_stats_table_rows(ui);

        self.temp_allocator.ui_stats_table_rows(ui, "temp memory");

        self.ping_pong_prev_set
            .allocator
            .ui_stats_table_rows(ui, "ping pong memory");

        ui.text("ping pong buffer");
        ui.next_column();
        ui.text(format!("{}", self.ping_pong_prev_set.buffers.len()));
        ui.next_column();

        ui.text("ping pong image");
        ui.next_column();
        ui.text(format!("{}", self.ping_pong_prev_set.images.len()));
        ui.next_column();
    }
}

/// State for a batch of draw calls, becomes a Vulkan sub-pass
#[derive(Clone, Copy)]
pub struct RenderState {
    pub color_output: ImageHandle,
    pub color_clear_value: [f32; 4],
    pub color_temp: Option<ImageHandle>,
    pub depth_temp: Option<ImageHandle>,
}

impl RenderState {
    pub fn new(color_output: ImageHandle, color_clear_value: &[f32; 4]) -> Self {
        Self {
            color_output,
            color_clear_value: *color_clear_value,
            color_temp: None,
            depth_temp: None,
        }
    }

    pub fn with_color_temp(mut self, image: ImageHandle) -> Self {
        self.color_temp = Some(image);
        self
    }

    pub fn with_depth_temp(mut self, image: ImageHandle) -> Self {
        self.depth_temp = Some(image);
        self
    }
}

#[derive(Clone, Copy)]
enum ParameterDesc {
    Buffer { handle: BufferHandle, usage: BufferUsage },
    Image { handle: ImageHandle, usage: ImageUsage },
}

struct CommandCommon {
    name: &'static CStr,
    params: RenderParameterDeclaration,
}

enum Command<'a> {
    Compute {
        common: CommandCommon,
        callback: Box<dyn FnOnce(RenderParameterAccess, vk::CommandBuffer) + 'a>,
    },
    Graphics {
        common: CommandCommon,
        state: RenderState,
        callback: Box<dyn FnOnce(RenderParameterAccess, vk::CommandBuffer, vk::RenderPass) + 'a>,
    },
}

struct SplitCommandBuffer {
    current: vk::CommandBuffer,
    next: Option<vk::CommandBuffer>,
    swap_image: Option<ImageHandle>,
}

impl SplitCommandBuffer {
    fn new(first: vk::CommandBuffer, second: vk::CommandBuffer, swap_image: Option<ImageHandle>) -> Self {
        Self {
            current: first,
            next: Some(second),
            swap_image,
        }
    }

    fn notify_image_use(&mut self, handle: ImageHandle, query_pool: &mut QueryPool) {
        if self.swap_image == Some(handle) {
            query_pool.end_command_buffer(self.current);
            self.current = self.next.take().unwrap();
            self.swap_image = None;
        }
    }
}

pub struct RenderParameterDeclaration(Vec<ParameterDesc>);

impl RenderParameterDeclaration {
    fn new() -> Self {
        Self(Vec::new())
    }

    pub fn add_buffer(&mut self, handle: BufferHandle, usage: BufferUsage) {
        self.0.push(ParameterDesc::Buffer { handle, usage });
    }

    pub fn add_image(&mut self, handle: ImageHandle, usage: ImageUsage) {
        self.0.push(ParameterDesc::Image { handle, usage })
    }
}

pub struct RenderParameterAccess<'a>(&'a RenderGraph);

impl<'a> RenderParameterAccess<'a> {
    fn new(render_graph: &'a RenderGraph) -> Self {
        Self(render_graph)
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> vk::Buffer {
        self.0.buffers.get(handle.0).unwrap().buffer().unwrap().0
    }

    pub fn get_image_view(&self, handle: ImageHandle) -> vk::ImageView {
        self.0.images.get(handle.0).unwrap().image_view().unwrap().0
    }
}

enum TemporaryHandle {
    Buffer(BufferHandle),
    Image(ImageHandle),
}

enum FinalUsage {
    Buffer { handle: BufferHandle, usage: BufferUsage },
    Image { handle: ImageHandle, usage: ImageUsage },
}

pub struct RenderSchedule<'a> {
    render_graph: &'a mut RenderGraph,
    temporaries: Vec<TemporaryHandle>,
    commands: Vec<Command<'a>>,
    final_usage: Vec<FinalUsage>,
}

impl<'a> RenderSchedule<'a> {
    pub fn new(render_graph: &'a mut RenderGraph) -> Self {
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

    pub fn create_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        allocator: &mut Allocator,
    ) -> BufferHandle {
        self.render_graph.create_buffer(desc, all_usage, allocator)
    }

    pub fn get_buffer_hack(&self, handle: BufferHandle) -> vk::Buffer {
        self.render_graph.buffers.get(handle.0).unwrap().buffer().unwrap().0
    }

    pub fn import_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
        final_usage: BufferUsage,
    ) -> BufferHandle {
        let handle = self
            .render_graph
            .create_ready_buffer(desc, all_usage, buffer, current_usage);
        self.render_graph.import_set.buffers.push(handle);
        self.final_usage.push(FinalUsage::Buffer {
            handle,
            usage: final_usage,
        });
        handle
    }

    pub fn describe_buffer(&mut self, desc: &BufferDesc) -> BufferHandle {
        let handle = BufferHandle(self.render_graph.buffers.allocate(BufferResource::Temporary {
            desc: *desc,
            all_usage: BufferUsage::empty(),
        }));
        self.temporaries.push(TemporaryHandle::Buffer(handle));
        handle
    }

    pub fn import_image(
        &mut self,
        desc: &ImageDesc,
        all_usage: ImageUsage,
        image: UniqueImage,
        current_usage: ImageUsage,
        final_usage: ImageUsage,
    ) -> ImageHandle {
        let handle = self
            .render_graph
            .create_ready_image(desc, all_usage, image, current_usage);
        self.render_graph.import_set.images.push(handle);
        self.final_usage.push(FinalUsage::Image {
            handle,
            usage: final_usage,
        });
        handle
    }

    pub fn describe_image(&mut self, desc: &ImageDesc) -> ImageHandle {
        let handle = ImageHandle(self.render_graph.images.allocate(ImageResource::Temporary {
            desc: *desc,
            all_usage: ImageUsage::empty(),
        }));
        self.temporaries.push(TemporaryHandle::Image(handle));
        handle
    }

    pub fn add_compute(
        &mut self,
        name: &'static CStr,
        decl: impl FnOnce(&mut RenderParameterDeclaration),
        callback: impl FnOnce(RenderParameterAccess, vk::CommandBuffer) + 'a,
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
        callback: impl FnOnce(RenderParameterAccess, vk::CommandBuffer, vk::RenderPass) + 'a,
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
        swap_image: Option<ImageHandle>,
        query_pool: &mut QueryPool,
    ) {
        // loop over commands to set usage for all resources
        for command in &self.commands {
            let common = match command {
                Command::Compute { common, .. } => common,
                Command::Graphics { common, state, .. } => {
                    self.render_graph
                        .images
                        .get_mut(state.color_output.0)
                        .unwrap()
                        .declare_usage(ImageUsage::COLOR_ATTACHMENT_WRITE);
                    if let Some(handle) = state.color_temp {
                        self.render_graph
                            .images
                            .get_mut(handle.0)
                            .unwrap()
                            .declare_usage(ImageUsage::TRANSIENT_COLOR_ATTACHMENT);
                    }
                    if let Some(handle) = state.depth_temp {
                        self.render_graph
                            .images
                            .get_mut(handle.0)
                            .unwrap()
                            .declare_usage(ImageUsage::TRANSIENT_DEPTH_ATTACHMENT)
                    }
                    common
                }
            };
            for parameter_desc in &common.params.0 {
                match parameter_desc {
                    ParameterDesc::Buffer { handle, usage } => {
                        self.render_graph
                            .buffers
                            .get_mut(handle.0)
                            .unwrap()
                            .declare_usage(*usage);
                    }
                    ParameterDesc::Image { handle, usage } => {
                        self.render_graph
                            .images
                            .get_mut(handle.0)
                            .unwrap()
                            .declare_usage(*usage);
                    }
                }
            }
        }

        // allocate temporaries
        for handle in &self.temporaries {
            match handle {
                TemporaryHandle::Buffer(handle) => self.render_graph.allocate_temporary_buffer(*handle),
                TemporaryHandle::Image(handle) => self.render_graph.allocate_temporary_image(*handle),
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
        let mut cmd = SplitCommandBuffer::new(pre_swapchain_cmd, post_swapchain_cmd, swap_image);
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

            for parameter_desc in &common.params.0 {
                match parameter_desc {
                    ParameterDesc::Buffer { handle, usage } => {
                        self.render_graph
                            .transition_buffer_usage(*handle, *usage, context, cmd.current);
                    }
                    ParameterDesc::Image { handle, usage } => {
                        self.render_graph
                            .transition_image_usage(*handle, *usage, context, &mut cmd, query_pool);
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
                    // TODO: initial layout as part of render pass (assumes UNDEFINED for now)
                    cmd.notify_image_use(state.color_output, query_pool);

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
                    let color_output_resource = self.render_graph.images.get(state.color_output.0).unwrap();
                    let color_output_desc = color_output_resource.desc();
                    let color_output_image_view = color_output_resource.image_view().unwrap();

                    // color temp (if present)
                    let (samples, color_temp_image_view) = if let Some(color_temp) = state.color_temp {
                        let color_temp_resource = self.render_graph.images.get(color_temp.0).unwrap();
                        let color_temp_desc = color_temp_resource.desc();
                        let color_temp_image_view = color_temp_resource.image_view().unwrap();

                        assert_eq!(color_output_desc.size(), color_temp_desc.size());
                        assert_eq!(color_output_desc.format, color_temp_desc.format);

                        clear_values.push(color_clear_value);

                        (color_temp_desc.samples, Some(color_temp_image_view))
                    } else {
                        (vk::SampleCountFlags::N1, None)
                    };

                    // depth temp (if present)
                    let (depth_format, depth_temp_image_view) = if let Some(depth_temp) = state.depth_temp {
                        let depth_temp_resource = self.render_graph.images.get(depth_temp.0).unwrap();
                        let depth_temp_desc = depth_temp_resource.desc();
                        let depth_temp_image_view = depth_temp_resource.image_view().unwrap();

                        assert_eq!(color_output_desc.size(), depth_temp_desc.size());
                        assert_eq!(samples, depth_temp_desc.samples);

                        clear_values.push(depth_clear_value);

                        (Some(depth_temp_desc.format), Some(depth_temp_image_view))
                    } else {
                        (None, None)
                    };

                    let render_pass =
                        self.render_graph
                            .render_cache
                            .get_render_pass(color_output_desc.format, depth_format, samples);

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

                    query_pool.emit_timestamp(cmd.current, timestamp_name);
                    (callback)(
                        RenderParameterAccess::new(self.render_graph),
                        cmd.current,
                        render_pass.0,
                    );

                    unsafe { context.device.cmd_end_render_pass(cmd.current) };

                    // TODO: final layout as part of render pass
                    self.render_graph
                        .images
                        .get_mut(state.color_output.0)
                        .unwrap()
                        .force_usage(ImageUsage::COLOR_ATTACHMENT_WRITE);
                }
            }

            if is_debug {
                unsafe { context.instance.cmd_end_debug_utils_label_ext(cmd.current) };
            }
        }

        // free temporaries
        for handle in self.temporaries.drain(..) {
            match handle {
                TemporaryHandle::Buffer(handle) => self.render_graph.free_buffer(handle),
                TemporaryHandle::Image(handle) => self.render_graph.free_image(handle),
            }
        }

        // emit any last barriers (usually at least the swap chain image)
        for final_usage in self.final_usage.drain(..) {
            match final_usage {
                FinalUsage::Buffer { handle, usage } => {
                    self.render_graph
                        .transition_buffer_usage(handle, usage, context, cmd.current);
                }
                FinalUsage::Image { handle, usage } => {
                    self.render_graph
                        .transition_image_usage(handle, usage, context, &mut cmd, query_pool);
                }
            }
        }
    }
}
