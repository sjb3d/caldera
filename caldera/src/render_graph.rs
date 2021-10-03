use crate::prelude::*;
use arrayvec::ArrayVec;
use imgui::Ui;
use slotmap::{new_key_type, SlotMap};
use spark::{vk, Builder, Device};
use std::{
    ffi::CStr,
    mem,
    sync::{Arc, Mutex},
};

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

new_key_type! {
    pub struct BufferId;

    pub struct ImageId;
}

pub(crate) enum BufferResource {
    Described {
        desc: BufferDesc,
        all_usage: BufferUsage,
    },
    Active {
        desc: BufferDesc,
        alloc: Option<Alloc>,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
        all_usage_check: BufferUsage,
    },
}

impl BufferResource {
    pub(crate) fn desc(&self) -> &BufferDesc {
        match self {
            BufferResource::Described { desc, .. } => desc,
            BufferResource::Active { desc, .. } => desc,
        }
    }

    pub(crate) fn buffer(&self) -> UniqueBuffer {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { buffer, .. } => *buffer,
        }
    }

    pub(crate) fn alloc(&self) -> Option<Alloc> {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { alloc, .. } => *alloc,
        }
    }

    fn declare_usage(&mut self, usage: BufferUsage) {
        match self {
            BufferResource::Described { ref mut all_usage, .. } => {
                *all_usage |= usage;
            }
            BufferResource::Active { all_usage_check, .. } => {
                if !all_usage_check.contains(usage) {
                    panic!("buffer usage {:?} was not declared in {:?}", usage, all_usage_check);
                }
            }
        }
    }

    pub(crate) fn transition_usage(&mut self, new_usage: BufferUsage, device: &Device, cmd: vk::CommandBuffer) {
        match self {
            BufferResource::Active {
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

pub(crate) enum ImageResource {
    Described {
        desc: ImageDesc,
        all_usage: ImageUsage,
    },
    Active {
        desc: ImageDesc,
        _alloc: Option<Alloc>,
        image: UniqueImage,
        image_view: UniqueImageView,
        current_usage: ImageUsage,
        all_usage_check: ImageUsage,
    },
}

impl ImageResource {
    pub(crate) fn desc(&self) -> &ImageDesc {
        match self {
            ImageResource::Described { desc, .. } => desc,
            ImageResource::Active { desc, .. } => desc,
        }
    }

    pub(crate) fn image(&self) -> UniqueImage {
        match self {
            ImageResource::Described { .. } => panic!("image is only described"),
            ImageResource::Active { image, .. } => *image,
        }
    }

    pub(crate) fn image_view(&self) -> UniqueImageView {
        match self {
            ImageResource::Described { .. } => panic!("image is only described"),
            ImageResource::Active { image_view, .. } => *image_view,
        }
    }

    fn declare_usage(&mut self, usage: ImageUsage) {
        match self {
            ImageResource::Described { ref mut all_usage, .. } => {
                *all_usage |= usage;
            }
            ImageResource::Active { all_usage_check, .. } => {
                if !all_usage_check.contains(usage) {
                    panic!("image usage {:?} was not declared in {:?}", usage, all_usage_check);
                }
            }
        }
    }

    pub(crate) fn transition_usage(&mut self, new_usage: ImageUsage, device: &Device, cmd: vk::CommandBuffer) {
        match self {
            ImageResource::Active {
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
            ImageResource::Active {
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

pub(crate) struct Resources {
    global_allocator: Allocator,
    resource_cache: ResourceCache,
    buffers: SlotMap<BufferId, BufferResource>,
    images: SlotMap<ImageId, ImageResource>,
}

impl Resources {
    pub(crate) fn new(context: &SharedContext, global_chunk_size: u32) -> Self {
        Self {
            global_allocator: Allocator::new(context, global_chunk_size),
            resource_cache: ResourceCache::new(context),
            buffers: SlotMap::with_key(),
            images: SlotMap::with_key(),
        }
    }

    fn add_buffer(
        &mut self,
        desc: &BufferDesc,
        alloc: Option<Alloc>,
        all_usage: BufferUsage,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
    ) -> BufferId {
        self.buffers.insert(BufferResource::Active {
            desc: *desc,
            alloc,
            buffer,
            current_usage,
            all_usage_check: all_usage,
        })
    }

    pub(crate) fn create_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        memory_property_flags: vk::MemoryPropertyFlags,
        allocator: Option<&mut Allocator>,
    ) -> BufferId {
        let all_usage_flags = all_usage.as_flags();
        let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
        let alloc = allocator
            .unwrap_or(&mut self.global_allocator)
            .allocate(&info.mem_req, memory_property_flags);
        let buffer = self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags);
        self.add_buffer(desc, Some(alloc), all_usage, buffer, BufferUsage::empty())
    }

    pub(crate) fn buffer_resource(&self, id: BufferId) -> &BufferResource {
        self.buffers.get(id).unwrap()
    }

    pub(crate) fn buffer_resource_mut(&mut self, id: BufferId) -> &mut BufferResource {
        self.buffers.get_mut(id).unwrap()
    }

    fn allocate_temporary_buffer(&mut self, id: BufferId, allocator: &mut Allocator) {
        let buffer_resource = self.buffers.get_mut(id).unwrap();
        *buffer_resource = match buffer_resource {
            BufferResource::Described { desc, all_usage } => {
                let all_usage_flags = all_usage.as_flags();
                let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
                let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
                let buffer = self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags);
                BufferResource::Active {
                    desc: *desc,
                    alloc: Some(alloc),
                    buffer,
                    current_usage: BufferUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("buffer is not temporary"),
        };
    }

    fn remove_buffer(&mut self, id: BufferId) {
        self.buffers.remove(id).unwrap();
    }

    fn add_image(
        &mut self,
        desc: &ImageDesc,
        alloc: Option<Alloc>,
        all_usage: ImageUsage,
        image: UniqueImage,
        current_usage: ImageUsage,
    ) -> ImageId {
        let image_view = self.resource_cache.get_image_view(desc, image);
        self.images.insert(ImageResource::Active {
            desc: *desc,
            _alloc: alloc,
            image,
            image_view,
            current_usage,
            all_usage_check: all_usage,
        })
    }

    pub fn create_image(
        &mut self,
        desc: &ImageDesc,
        all_usage: ImageUsage,
        allocator: Option<&mut Allocator>,
    ) -> ImageId {
        let all_usage_flags = all_usage.as_flags();
        let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let info = self.resource_cache.get_image_info(desc, all_usage_flags);
        let alloc = allocator
            .unwrap_or(&mut self.global_allocator)
            .allocate(&info.mem_req, memory_property_flags);
        let image = self.resource_cache.get_image(desc, &info, &alloc, all_usage_flags);
        self.add_image(desc, Some(alloc), all_usage, image, ImageUsage::empty())
    }

    pub(crate) fn image_resource(&self, id: ImageId) -> &ImageResource {
        self.images.get(id).unwrap()
    }

    pub(crate) fn image_resource_mut(&mut self, id: ImageId) -> &mut ImageResource {
        self.images.get_mut(id).unwrap()
    }

    fn allocate_temporary_image(&mut self, id: ImageId, allocator: &mut Allocator) {
        let image_resource = self.images.get_mut(id).unwrap();
        *image_resource = match image_resource {
            ImageResource::Described { desc, all_usage } => {
                let all_usage_flags = all_usage.as_flags();
                let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                let info = self.resource_cache.get_image_info(desc, all_usage_flags);
                let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
                let image = self.resource_cache.get_image(desc, &info, &alloc, all_usage_flags);
                let image_view = self.resource_cache.get_image_view(desc, image);
                ImageResource::Active {
                    desc: *desc,
                    _alloc: Some(alloc),
                    image,
                    image_view,
                    current_usage: ImageUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("image is not temporary"),
        };
    }

    fn remove_image(&mut self, id: ImageId) {
        self.images.remove(id).unwrap();
    }

    fn transition_buffer_usage(
        &mut self,
        id: BufferId,
        new_usage: BufferUsage,
        context: &Context,
        cmd: vk::CommandBuffer,
    ) {
        self.buffers
            .get_mut(id)
            .unwrap()
            .transition_usage(new_usage, &context.device, cmd);
    }

    fn transition_image_usage(
        &mut self,
        id: ImageId,
        new_usage: ImageUsage,
        context: &Context,
        cmd: &mut SplitCommandBuffer,
        query_pool: &mut QueryPool,
    ) {
        cmd.notify_image_use(id, query_pool);
        self.images
            .get_mut(id)
            .unwrap()
            .transition_usage(new_usage, &context.device, cmd.current);
    }
}

pub struct RenderGraph {
    resources: Arc<Mutex<Resources>>,
    render_cache: RenderCache,
    temp_allocator: Allocator,
    ping_pong_current_set: ResourceSet,
    ping_pong_prev_set: ResourceSet,
    import_set: ResourceSet,
}

impl RenderGraph {
    pub fn new(
        context: &SharedContext,
        global_chunk_size: u32,
        temp_chunk_size: u32,
        ping_pong_chunk_size: u32,
    ) -> Self {
        Self {
            resources: Arc::new(Mutex::new(Resources::new(context, global_chunk_size))),
            render_cache: RenderCache::new(context),
            temp_allocator: Allocator::new(context, temp_chunk_size),
            ping_pong_current_set: ResourceSet::new(context, ping_pong_chunk_size),
            ping_pong_prev_set: ResourceSet::new(context, ping_pong_chunk_size),
            import_set: ResourceSet::new(context, 0),
        }
    }

    pub(crate) fn resources_hack(&self) -> &Arc<Mutex<Resources>> {
        &self.resources
    }

    pub fn create_buffer(&mut self, desc: &BufferDesc, all_usage: BufferUsage) -> BufferId {
        self.resources
            .lock()
            .unwrap()
            .create_buffer(desc, all_usage, vk::MemoryPropertyFlags::DEVICE_LOCAL, None)
    }

    pub fn get_buffer_desc(&self, id: BufferId) -> BufferDesc {
        *self.resources.lock().unwrap().buffers.get(id).unwrap().desc()
    }

    pub fn create_image(&mut self, desc: &ImageDesc, all_usage: ImageUsage) -> ImageId {
        self.resources.lock().unwrap().create_image(desc, all_usage, None)
    }

    pub fn get_image_desc(&self, id: ImageId) -> ImageDesc {
        *self.resources.lock().unwrap().images.get(id).unwrap().desc()
    }

    pub fn begin_frame(&mut self) {
        mem::swap(&mut self.ping_pong_current_set, &mut self.ping_pong_prev_set);
        let mut resources = self.resources.lock().unwrap();
        self.ping_pong_current_set.begin_frame(&mut resources);
        self.import_set.begin_frame(&mut resources);
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        let resources = self.resources.lock().unwrap();

        resources.global_allocator.ui_stats_table_rows(ui, "global memory");

        ui.text("graph buffers");
        ui.next_column();
        ui.text(format!("{}", resources.buffers.len()));
        ui.next_column();

        ui.text("graph images");
        ui.next_column();
        ui.text(format!("{}", resources.images.len()));
        ui.next_column();

        resources.resource_cache.ui_stats_table_rows(ui, "graph");
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

    pub fn get_buffer_hack(&self, id: BufferId) -> vk::Buffer {
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
                .add_buffer(desc, None, all_usage, buffer, current_usage);
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
        let buffer_id = self
            .render_graph
            .resources
            .lock()
            .unwrap()
            .buffers
            .insert(BufferResource::Described {
                desc: *desc,
                all_usage: BufferUsage::empty(),
            });
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
        let image_id =
            self.render_graph
                .resources
                .lock()
                .unwrap()
                .add_image(desc, None, all_usage, image, current_usage);
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
        let image_id = self
            .render_graph
            .resources
            .lock()
            .unwrap()
            .images
            .insert(ImageResource::Described {
                desc: *desc,
                all_usage: ImageUsage::empty(),
            });
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
                            .images
                            .get_mut(state.color_output_id)
                            .unwrap()
                            .declare_usage(ImageUsage::COLOR_ATTACHMENT_WRITE);
                        if let Some(id) = state.color_temp_id {
                            resources
                                .images
                                .get_mut(id)
                                .unwrap()
                                .declare_usage(ImageUsage::TRANSIENT_COLOR_ATTACHMENT);
                        }
                        if let Some(id) = state.depth_temp_id {
                            resources
                                .images
                                .get_mut(id)
                                .unwrap()
                                .declare_usage(ImageUsage::TRANSIENT_DEPTH_ATTACHMENT)
                        }
                        common
                    }
                };
                for parameter_desc in &common.params.0 {
                    match parameter_desc {
                        ParameterDesc::Buffer { id, usage } => {
                            resources.buffers.get_mut(*id).unwrap().declare_usage(*usage);
                        }
                        ParameterDesc::Image { id, usage } => {
                            resources.images.get_mut(*id).unwrap().declare_usage(*usage);
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
                            resources.transition_image_usage(*id, *usage, context, &mut cmd, query_pool);
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
                        let color_output_resource = resources.images.get(state.color_output_id).unwrap();
                        let color_output_desc = color_output_resource.desc();
                        let color_output_image_view = color_output_resource.image_view();

                        // color temp (if present)
                        let (samples, color_temp_image_view) = if let Some(color_temp_id) = state.color_temp_id {
                            let color_temp_resource = resources.images.get(color_temp_id).unwrap();
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
                            let depth_temp_resource = resources.images.get(depth_temp_id).unwrap();
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
                        .images
                        .get_mut(state.color_output_id)
                        .unwrap()
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
                        resources.transition_image_usage(id, usage, context, &mut cmd, query_pool);
                    }
                }
            }
        }
    }
}
