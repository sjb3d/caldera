use crate::{heap::*, prelude::*};
use bytemuck::Pod;
use slotmap::{new_key_type, SlotMap};
use spark::vk;
use std::{
    collections::VecDeque,
    future::Future,
    mem,
    pin::Pin,
    slice,
    sync::{Arc, Mutex},
    task::{Context as PollCtx, Poll, Waker},
    thread,
};
use tokio::{
    runtime::Builder,
    sync::{mpsc, oneshot},
};

type BackgroundTask = Pin<Box<dyn Future<Output = ()> + Send>>;

pub struct BackgroundTaskSystem {
    send: mpsc::Sender<BackgroundTask>,
}

impl BackgroundTaskSystem {
    pub fn new() -> Self {
        let (send, mut recv) = mpsc::channel(1);
        thread::spawn({
            move || {
                let rt = Builder::new_current_thread().enable_all().build().unwrap();
                rt.block_on(async {
                    while let Some(task) = recv.recv().await {
                        rt.spawn(task);
                    }
                });
            }
        });
        Self { send }
    }

    pub fn spawn_task<F, T>(&self, f: F) -> TaskOutput<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let task = async move {
            let output = f.await;
            tx.send(output)
                .unwrap_or_else(|_e| panic!("failed to send output to receiver"));
        };
        self.send
            .blocking_send(Box::pin(task))
            .unwrap_or_else(|_e| panic!("failed to send task to runner"));
        TaskOutput::Waiting(rx)
    }
}

pub enum TaskOutput<T> {
    Waiting(oneshot::Receiver<T>),
    Received(T),
}

impl<T> TaskOutput<T> {
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if let Self::Waiting(rx) = self {
            match rx.try_recv() {
                Ok(t) => *self = Self::Received(t),
                Err(oneshot::error::TryRecvError::Closed) => panic!(),
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
        }
        if let Self::Received(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn get(&mut self) -> Option<&T> {
        self.get_mut().map(|value| &*value)
    }
}

struct StagingMapping(*mut u8);

unsafe impl Send for StagingMapping {}
unsafe impl Sync for StagingMapping {}

new_key_type! {
    struct StagingAllocId;
}

enum StagingAllocState {
    Pending { waker: Option<Waker> },
    Done { offset: u32 },
}

struct StagingAlloc {
    size: u32,
    alignment: u32,
    state: StagingAllocState,
}

#[derive(Clone, Copy)]
struct StagingRegion {
    offset: u32,
    size: u32,
}

struct StagingDesc {
    _buffer_id: BufferId,
    buffer: vk::Buffer,
    mapping: StagingMapping,
    alignment: u32,
}

impl StagingDesc {
    fn new(context: &SharedContext, resources: &SharedResources, size: u32) -> Self {
        let mut resources_lock = resources.lock().unwrap();

        let buffer_desc = BufferDesc::new(size as usize);
        let buffer_id = resources_lock.create_buffer(
            &buffer_desc,
            BufferUsage::TRANSFER_READ,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let buffer_resource = resources_lock.buffer_resource(buffer_id);
        let buffer = buffer_resource.buffer().0;
        let mapping = StagingMapping({
            let alloc = buffer_resource.alloc().unwrap();
            unsafe {
                context
                    .device
                    .map_memory(alloc.mem, alloc.offset, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .unwrap() as *mut u8
        });

        let alignment = context
            .physical_device_properties
            .limits
            .min_storage_buffer_offset_alignment as u32;

        Self {
            _buffer_id: buffer_id,
            buffer,
            mapping,
            alignment,
        }
    }
}

struct StagingState {
    heap: HeapAllocator,
    allocs: SlotMap<StagingAllocId, StagingAlloc>,
    queue: VecDeque<StagingAllocId>,
}

impl StagingState {
    fn new(staging_size: u32) -> Self {
        Self {
            heap: HeapAllocator::new(staging_size),
            allocs: SlotMap::with_key(),
            queue: VecDeque::new(),
        }
    }

    fn alloc(&mut self, size: u32, alignment: u32) -> StagingAllocId {
        let id = self.allocs.insert(StagingAlloc {
            size,
            alignment,
            state: StagingAllocState::Pending { waker: None },
        });
        self.queue.push_back(id);
        self.process_queue();
        id
    }

    fn poll_alloc(&mut self, id: StagingAllocId, cx: &mut PollCtx<'_>) -> Poll<u32> {
        let alloc = self.allocs.get_mut(id).unwrap();
        if let StagingAllocState::Done { offset } = alloc.state {
            self.allocs.remove(id);
            Poll::Ready(offset)
        } else {
            alloc.state = StagingAllocState::Pending {
                waker: Some(cx.waker().clone()),
            };
            Poll::Pending
        }
    }

    fn process_queue(&mut self) {
        while let Some(id) = self.queue.pop_front() {
            let alloc = &mut self.allocs[id];
            if let Some(offset) = self.heap.alloc(alloc.size, alloc.alignment) {
                let mut state = StagingAllocState::Done { offset };
                mem::swap(&mut state, &mut alloc.state);
                match state {
                    StagingAllocState::Pending { mut waker } => {
                        if let Some(waker) = waker.take() {
                            waker.wake();
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                self.queue.push_front(id);
                break;
            }
        }
    }

    fn free(&mut self, offset: u32) {
        self.heap.free(offset);
        self.process_queue();
    }
}

struct ResourceStagingOffsetFuture {
    shared: Arc<ResourceLoaderShared>,
    id: StagingAllocId,
}

impl Future for ResourceStagingOffsetFuture {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, cx: &mut PollCtx<'_>) -> Poll<Self::Output> {
        self.shared.staging_state.lock().unwrap().poll_alloc(self.id, cx)
    }
}

enum TransferTask {
    Buffer {
        staging_offset: u32,
        buffer_id: BufferId,
        initial_usage: BufferUsage,
        sender: oneshot::Sender<BufferId>,
    },
    Image {
        staging_offset: u32,
        image_id: ImageId,
        initial_usage: ImageUsage,
        sender: oneshot::Sender<ImageId>,
    },
}

struct FreeTask {
    staging_offset: u32,
    countdown: u32,
}

struct TransferState {
    transfers: VecDeque<TransferTask>,
    frees: Vec<FreeTask>,
}

impl TransferState {
    fn new() -> Self {
        Self {
            transfers: VecDeque::new(),
            frees: Vec::new(),
        }
    }
}

pub struct GraphicsTaskContext<'ctx, 'graph> {
    pub schedule: &'ctx mut RenderSchedule<'graph>,
    pub context: &'graph Context,
    pub descriptor_pool: &'graph DescriptorPool,
    pub pipeline_cache: &'graph PipelineCache,
}

struct GraphicsState {
    tasks: VecDeque<Box<dyn FnOnce(GraphicsTaskContext) + Send + 'static>>,
}

impl GraphicsState {
    fn new() -> Self {
        Self { tasks: VecDeque::new() }
    }
}

struct ResourceLoaderShared {
    context: SharedContext,
    resources: SharedResources,
    staging_desc: StagingDesc,
    staging_state: Mutex<StagingState>,
    transfer_state: Mutex<TransferState>,
    graphics_state: Mutex<GraphicsState>,
}

impl ResourceLoaderShared {
    unsafe fn staging_mapping(&self, region: StagingRegion, offset: usize, len: usize) -> &'_ mut [u8] {
        if (offset + len) > (region.size as usize) {
            panic!("mapping region out of bounds");
        }
        slice::from_raw_parts_mut(self.staging_desc.mapping.0.add(region.offset as usize + offset), len)
    }

    fn transfer_buffer(
        &self,
        staging_offset: u32,
        buffer_id: BufferId,
        initial_usage: BufferUsage,
    ) -> impl Future<Output = BufferId> {
        let (tx, rx) = oneshot::channel();
        self.transfer_state
            .lock()
            .unwrap()
            .transfers
            .push_back(TransferTask::Buffer {
                staging_offset,
                buffer_id,
                initial_usage,
                sender: tx,
            });
        async { rx.await.unwrap() }
    }

    fn transfer_image(
        &self,
        staging_offset: u32,
        image_id: ImageId,
        initial_usage: ImageUsage,
    ) -> impl Future<Output = ImageId> {
        let (tx, rx) = oneshot::channel();
        self.transfer_state
            .lock()
            .unwrap()
            .transfers
            .push_back(TransferTask::Image {
                staging_offset,
                image_id,
                initial_usage,
                sender: tx,
            });
        async { rx.await.unwrap() }
    }
}

#[derive(Clone)]
pub struct ResourceLoader {
    shared: Arc<ResourceLoaderShared>,
}

impl ResourceLoader {
    pub(crate) fn new(context: &SharedContext, resources: &SharedResources, staging_size: u32) -> Self {
        Self {
            shared: Arc::new(ResourceLoaderShared {
                context: SharedContext::clone(context),
                resources: SharedResources::clone(resources),
                staging_desc: StagingDesc::new(context, resources, staging_size),
                staging_state: Mutex::new(StagingState::new(staging_size)),
                transfer_state: Mutex::new(TransferState::new()),
                graphics_state: Mutex::new(GraphicsState::new()),
            }),
        }
    }

    pub(crate) fn begin_frame(&self, cmd: vk::CommandBuffer) {
        let mut transfer_state = self.shared.transfer_state.lock().unwrap();

        let mut has_free = false;
        for free in transfer_state.frees.iter_mut() {
            free.countdown -= 1;
            if free.countdown == 0 {
                has_free = true;
            }
        }
        if has_free {
            let mut staging_state = self.shared.staging_state.lock().unwrap();
            transfer_state.frees.retain(|free| {
                if free.countdown == 0 {
                    staging_state.free(free.staging_offset);
                    false
                } else {
                    true
                }
            });
        }

        let device = &self.shared.context.device;
        let mut resources = self.shared.resources.lock().unwrap();
        while let Some(transfer) = transfer_state.transfers.pop_front() {
            let staging_offset = match transfer {
                TransferTask::Buffer {
                    staging_offset,
                    buffer_id,
                    initial_usage,
                    sender,
                } => {
                    let buffer_resource = resources.buffer_resource_mut(buffer_id);

                    buffer_resource.transition_usage(BufferUsage::TRANSFER_WRITE, device, cmd);

                    let desc = buffer_resource.desc();
                    let region = vk::BufferCopy {
                        src_offset: staging_offset as vk::DeviceSize,
                        dst_offset: 0,
                        size: desc.size as vk::DeviceSize,
                    };
                    unsafe {
                        device.cmd_copy_buffer(
                            cmd,
                            self.shared.staging_desc.buffer,
                            buffer_resource.buffer().0,
                            slice::from_ref(&region),
                        )
                    };

                    buffer_resource.transition_usage(initial_usage, device, cmd);
                    sender.send(buffer_id).unwrap();
                    staging_offset
                }
                TransferTask::Image {
                    staging_offset,
                    image_id,
                    initial_usage,
                    sender,
                } => {
                    let image_resource = resources.image_resource_mut(image_id);

                    image_resource.transition_usage(ImageUsage::TRANSFER_WRITE, device, cmd);

                    let desc = image_resource.desc();

                    let bits_per_elements = desc.first_format().bits_per_element();
                    let layer_count = desc.layer_count_or_zero.max(1) as usize;
                    let mut mip_width = desc.width as usize;
                    let mut mip_height = desc.height_or_zero.max(1) as usize;
                    let mut mip_offset = 0;
                    for mip_index in 0..desc.mip_count {
                        let region = vk::BufferImageCopy {
                            buffer_offset: ((staging_offset as usize) + mip_offset) as vk::DeviceSize,
                            buffer_row_length: mip_width as u32,
                            buffer_image_height: mip_height as u32,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: desc.aspect_mask,
                                mip_level: mip_index as u32,
                                base_array_layer: 0,
                                layer_count: layer_count as u32,
                            },
                            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                            image_extent: vk::Extent3D {
                                width: mip_width as u32,
                                height: mip_height as u32,
                                depth: 1,
                            },
                        };
                        unsafe {
                            device.cmd_copy_buffer_to_image(
                                cmd,
                                self.shared.staging_desc.buffer,
                                image_resource.image().0,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                slice::from_ref(&region),
                            )
                        };
                        let mip_layer_size = (mip_width * mip_height * bits_per_elements) / 8;
                        mip_offset += mip_layer_size * layer_count;
                        mip_width /= 2;
                        mip_height /= 2;
                    }

                    image_resource.transition_usage(initial_usage, device, cmd);
                    sender.send(image_id).unwrap();
                    staging_offset
                }
            };
            transfer_state.frees.push(FreeTask {
                staging_offset,
                countdown: CommandBufferPool::COUNT as u32,
            });
        }
    }

    pub fn context(&self) -> SharedContext {
        Arc::clone(&self.shared.context)
    }

    pub fn bindless_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.shared.resources.lock().unwrap().bindless_descriptor_set_layout()
    }

    pub fn begin_schedule<'graph>(
        &self,
        render_graph: &'graph mut RenderGraph,
        context: &'graph Context,
        descriptor_pool: &'graph DescriptorPool,
        pipeline_cache: &'graph PipelineCache,
    ) -> RenderSchedule<'graph> {
        let mut schedule = RenderSchedule::new(render_graph);

        let mut graphics_state = self.shared.graphics_state.lock().unwrap();
        while let Some(task) = graphics_state.tasks.pop_front() {
            task(GraphicsTaskContext {
                schedule: &mut schedule,
                context,
                descriptor_pool,
                pipeline_cache,
            });
        }

        schedule
    }

    pub fn buffer_writer(&self, desc: &BufferDesc, all_usage: BufferUsage) -> impl Future<Output = BufferWriter> {
        let buffer_id = self.shared.resources.lock().unwrap().create_buffer(
            desc,
            all_usage | BufferUsage::TRANSFER_WRITE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let size = desc.size as u32;
        let offset = ResourceStagingOffsetFuture {
            shared: Arc::clone(&self.shared),
            id: self
                .shared
                .staging_state
                .lock()
                .unwrap()
                .alloc(size, self.shared.staging_desc.alignment),
        };

        let shared = Arc::clone(&self.shared);
        async move {
            BufferWriter {
                shared,
                buffer_id,
                initial_usage: all_usage,
                staging_region: StagingRegion {
                    offset: offset.await,
                    size,
                },
                write_offset: 0,
            }
        }
    }

    pub fn get_buffer(&self, id: BufferId) -> vk::Buffer {
        self.shared.resources.lock().unwrap().buffer_resource(id).buffer().0
    }

    pub fn get_buffer_accel(&self, id: BufferId) -> vk::AccelerationStructureKHR {
        self.shared
            .resources
            .lock()
            .unwrap()
            .buffer_resource(id)
            .accel()
            .unwrap()
    }

    pub fn get_buffer_bindless_id(&self, id: BufferId) -> BindlessId {
        self.shared
            .resources
            .lock()
            .unwrap()
            .buffer_resource(id)
            .bindless_id()
            .unwrap()
    }

    pub fn image_writer(&self, desc: &ImageDesc, all_usage: ImageUsage) -> impl Future<Output = ImageWriter> {
        let image_id = self
            .shared
            .resources
            .lock()
            .unwrap()
            .create_image(desc, all_usage | ImageUsage::TRANSFER_WRITE);

        let size = desc.staging_size() as u32;
        let offset = ResourceStagingOffsetFuture {
            shared: Arc::clone(&self.shared),
            id: self
                .shared
                .staging_state
                .lock()
                .unwrap()
                .alloc(size, self.shared.staging_desc.alignment),
        };

        let shared = Arc::clone(&self.shared);
        async move {
            ImageWriter {
                shared,
                image_id,
                initial_usage: all_usage,
                staging_region: StagingRegion {
                    offset: offset.await,
                    size,
                },
                write_offset: 0,
            }
        }
    }

    pub fn get_image_view(&self, id: ImageId, view_desc: ImageViewDesc) -> vk::ImageView {
        self.shared.resources.lock().unwrap().image_view(id, view_desc).0
    }

    pub fn get_image_bindless_id(&self, id: ImageId) -> BindlessId {
        self.shared
            .resources
            .lock()
            .unwrap()
            .image_resource(id)
            .bindless_id()
            .unwrap()
    }

    pub fn graphics<F, T>(&self, func: F) -> impl Future<Output = T>
    where
        F: FnOnce(GraphicsTaskContext) -> T + Send + 'static,
        T: Send + 'static,
    {
        let mut graphics_state = self.shared.graphics_state.lock().unwrap();
        let (tx, rx) = oneshot::channel();
        graphics_state.tasks.push_back(Box::new(move |ctx| {
            tx.send(func(ctx))
                .unwrap_or_else(|_| panic!("failed to send remote call result"))
        }));
        async { rx.await.unwrap() }
    }
}

pub fn spawn<F>(fut: F) -> SpawnResult<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    SpawnResult(tokio::spawn(fut))
}

pub struct SpawnResult<T>(tokio::task::JoinHandle<T>);

impl<T> Future for SpawnResult<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, cx: &mut PollCtx<'_>) -> Poll<Self::Output> {
        let fut = unsafe { self.map_unchecked_mut(|r| &mut r.0) };
        match fut.poll(cx) {
            Poll::Ready(t) => Poll::Ready(t.unwrap()),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl<T: Pod> AsBytes for T {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl<T: Pod> AsBytes for [T] {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
}

pub trait StagingWriter {
    fn write<T: AsBytes + ?Sized>(&mut self, pod: &T);
    fn written(&self) -> usize;
    fn write_zeros(&mut self, len: usize);
}

macro_rules! writer_impl {
    ($name:ident) => {
        impl StagingWriter for $name {
            fn write<T: AsBytes + ?Sized>(&mut self, pod: &T) {
                let src_bytes = pod.as_bytes();
                let dst_bytes = unsafe {
                    self.shared
                        .staging_mapping(self.staging_region, self.write_offset, src_bytes.len())
                };
                dst_bytes.copy_from_slice(src_bytes);
                self.write_offset += src_bytes.len();
            }

            fn written(&self) -> usize {
                self.write_offset
            }

            fn write_zeros(&mut self, len: usize) {
                let dst_bytes = unsafe {
                    self.shared
                        .staging_mapping(self.staging_region, self.write_offset, len)
                };
                for dst in dst_bytes.iter_mut() {
                    *dst = 0;
                }
                self.write_offset += len;
            }
        }
    };
}

pub struct BufferWriter {
    shared: Arc<ResourceLoaderShared>,
    staging_region: StagingRegion,
    write_offset: usize,
    buffer_id: BufferId,
    initial_usage: BufferUsage,
}

impl BufferWriter {
    pub fn finish(mut self) -> impl Future<Output = BufferId> {
        self.write_zeros(self.staging_region.size as usize - self.write_offset);
        self.shared
            .transfer_buffer(self.staging_region.offset, self.buffer_id, self.initial_usage)
    }
}

writer_impl!(BufferWriter);

pub struct ImageWriter {
    shared: Arc<ResourceLoaderShared>,
    staging_region: StagingRegion,
    write_offset: usize,
    image_id: ImageId,
    initial_usage: ImageUsage,
}

impl ImageWriter {
    pub fn finish(mut self) -> impl Future<Output = ImageId> {
        self.write_zeros(self.staging_region.size as usize - self.write_offset);
        self.shared
            .transfer_image(self.staging_region.offset, self.image_id, self.initial_usage)
    }
}

writer_impl!(ImageWriter);
