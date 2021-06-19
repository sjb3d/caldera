use crate::{heap::*, prelude::*, resource::*};
use bytemuck::Pod;
use imgui::Ui;
use spark::vk;
use std::collections::VecDeque;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::thread::JoinHandle;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StaticBufferHandle(ResourceHandle);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StaticImageHandle(ResourceHandle);

enum StaticBufferResource {
    Empty,
    Mapped,
    Ready { desc: BufferDesc, buffer: UniqueBuffer },
}

enum StaticImageResource {
    Empty,
    Mapped,
    Ready {
        desc: ImageDesc,
        image: UniqueImage,
        image_view: UniqueImageView,
    },
}

struct ResourceStagingWrite {
    loader: Box<dyn FnOnce(&mut ResourceAllocator) + Send + 'static>,
}

#[derive(Clone, Copy)]
enum ResourceStagingTransferResource {
    Buffer {
        handle: StaticBufferHandle,
        desc: BufferDesc,
        all_usage: BufferUsage,
    },
    Image {
        handle: StaticImageHandle,
        desc: ImageDesc,
        all_usage: ImageUsage,
    },
}

#[derive(Clone, Copy)]
struct ResourceStagingTransfer {
    resource: ResourceStagingTransferResource,
    staging_offset: u32,
}

struct ResourceStagingFree {
    staging_offset: u32,
    countdown: u32,
}

struct ResourceStagingMapping(*mut u8);

unsafe impl Send for ResourceStagingMapping {}
unsafe impl Sync for ResourceStagingMapping {}

struct ResourceLoaderShared {
    context: Arc<Context>,
    exit_signal: AtomicBool,
    buffers: Mutex<ResourceVec<StaticBufferResource>>,
    images: Mutex<ResourceVec<StaticImageResource>>,
    staging_buffer: UniqueBuffer,
    staging_mapping: ResourceStagingMapping,
    staging_size: u32,
    heap: Mutex<HeapAllocator>,
    heap_freed: Condvar,
    writes: Mutex<VecDeque<ResourceStagingWrite>>,
    write_added: Condvar,
    transfers: Mutex<VecDeque<ResourceStagingTransfer>>,
}

impl ResourceLoaderShared {
    fn allocate_staging(&self, size: u32, alignment: u32) -> Option<u32> {
        if size > self.staging_size {
            panic!("allocation size {} too large for staging area", size);
        }
        let mut heap = self.heap.lock().unwrap();
        loop {
            if self.should_exit() {
                break None;
            } else if let Some(staging_offset) = heap.alloc(size, alignment) {
                break Some(staging_offset);
            } else {
                heap = self.heap_freed.wait(heap).unwrap();
            }
        }
    }

    fn pop_write(&self) -> Option<ResourceStagingWrite> {
        let mut writes = self.writes.lock().unwrap();
        loop {
            if self.should_exit() {
                break None;
            } else if let Some(write) = writes.pop_front() {
                break Some(write);
            } else {
                writes = self.write_added.wait(writes).unwrap();
            }
        }
    }

    fn try_pop_transfer(&self) -> Option<ResourceStagingTransfer> {
        let mut transfers = self.transfers.lock().unwrap();
        transfers.pop_front()
    }

    fn should_exit(&self) -> bool {
        self.exit_signal.load(Ordering::Acquire)
    }
}

pub struct ResourceLoader {
    shared: Arc<ResourceLoaderShared>,
    resource_cache: ResourceCache,
    frees: Vec<ResourceStagingFree>,
    writer_thread_join: Option<JoinHandle<()>>,
}

impl ResourceLoader {
    pub fn new(context: &Arc<Context>, allocator: &mut Allocator, staging_size: u32) -> Self {
        let mut resource_cache = ResourceCache::new(context);

        let (staging_buffer, staging_mapping) = {
            let desc = BufferDesc {
                size: staging_size as usize,
            };
            let all_usage_flags = vk::BufferUsageFlags::TRANSFER_SRC;
            let memory_property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let info = resource_cache.get_buffer_info(&desc, all_usage_flags);
            let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
            let buffer = resource_cache.get_buffer(&desc, &info, &alloc, all_usage_flags);
            let mapping = unsafe {
                context
                    .device
                    .map_memory(alloc.mem, alloc.offset, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .unwrap() as *mut u8;
            (buffer, mapping)
        };

        let shared = Arc::new(ResourceLoaderShared {
            context: Arc::clone(&context),
            exit_signal: AtomicBool::new(false),
            buffers: Mutex::new(ResourceVec::new()),
            images: Mutex::new(ResourceVec::new()),
            staging_buffer,
            staging_mapping: ResourceStagingMapping(staging_mapping),
            staging_size,
            heap: Mutex::new(HeapAllocator::new(staging_size)),
            heap_freed: Condvar::new(),
            writes: Mutex::new(VecDeque::new()),
            write_added: Condvar::new(),
            transfers: Mutex::new(VecDeque::new()),
        });

        let writer_thread_join = Some(thread::spawn({
            let mut allocator = ResourceAllocator {
                shared: Arc::clone(&shared),
            };
            move || {
                while let Some(write) = allocator.shared.pop_write() {
                    (write.loader)(&mut allocator);
                }
            }
        }));

        Self {
            shared,
            resource_cache,
            frees: Vec::new(),
            writer_thread_join,
        }
    }

    pub fn create_buffer(&mut self) -> StaticBufferHandle {
        let mut buffers = self.shared.buffers.lock().unwrap();
        StaticBufferHandle(buffers.allocate(StaticBufferResource::Empty))
    }

    pub fn create_image(&mut self) -> StaticImageHandle {
        let mut images = self.shared.images.lock().unwrap();
        StaticImageHandle(images.allocate(StaticImageResource::Empty))
    }

    pub fn async_load(&mut self, loader: impl FnOnce(&mut ResourceAllocator) + Send + 'static) {
        let mut writes = self.shared.writes.lock().unwrap();
        writes.push_back(ResourceStagingWrite {
            loader: Box::new(loader),
        });
        self.shared.write_added.notify_one();
    }

    pub fn process_frees(&mut self) {
        let mut has_free = false;
        for free in self.frees.iter_mut() {
            free.countdown -= 1;
            if free.countdown == 0 {
                has_free = true;
            }
        }
        if has_free {
            let mut heap = self.shared.heap.lock().unwrap();
            self.frees.retain(|free| {
                if free.countdown == 0 {
                    heap.free(free.staging_offset);
                    false
                } else {
                    true
                }
            });
            self.shared.heap_freed.notify_one();
        }
    }

    pub fn begin_frame(&mut self, allocator: &mut Allocator, cmd: vk::CommandBuffer) {
        self.process_frees();

        while let Some(transfer) = self.shared.try_pop_transfer() {
            let staging_offset = transfer.staging_offset;
            match transfer.resource {
                ResourceStagingTransferResource::Buffer {
                    handle,
                    desc,
                    all_usage,
                } => {
                    let buffer = {
                        let all_usage_flags = all_usage.as_flags() | vk::BufferUsageFlags::TRANSFER_DST;
                        let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                        let info = self.resource_cache.get_buffer_info(&desc, all_usage_flags);
                        let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
                        self.resource_cache.get_buffer(&desc, &info, &alloc, all_usage_flags)
                    };

                    let region = vk::BufferCopy {
                        src_offset: staging_offset as vk::DeviceSize,
                        dst_offset: 0,
                        size: desc.size as vk::DeviceSize,
                    };

                    unsafe {
                        self.shared.context.device.cmd_copy_buffer(
                            cmd,
                            self.shared.staging_buffer.0,
                            buffer.0,
                            slice::from_ref(&region),
                        )
                    };

                    emit_buffer_barrier(
                        BufferUsage::TRANSFER_WRITE,
                        all_usage,
                        buffer.0,
                        &self.shared.context.device,
                        cmd,
                    );

                    let mut buffers = self.shared.buffers.lock().unwrap();
                    let resource = buffers.get_mut(handle.0).unwrap();
                    assert!(matches!(resource, StaticBufferResource::Mapped));
                    *resource = StaticBufferResource::Ready { desc, buffer };
                }
                ResourceStagingTransferResource::Image {
                    handle,
                    desc,
                    all_usage,
                } => {
                    let image = {
                        let all_usage_flags = all_usage.as_flags() | vk::ImageUsageFlags::TRANSFER_DST;
                        let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                        let info = self.resource_cache.get_image_info(&desc, all_usage_flags);
                        let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
                        self.resource_cache.get_image(&desc, &info, &alloc, all_usage_flags)
                    };

                    emit_image_barrier(
                        ImageUsage::empty(),
                        ImageUsage::TRANSFER_WRITE,
                        image.0,
                        desc.aspect_mask,
                        &self.shared.context.device,
                        cmd,
                    );

                    let region = vk::BufferImageCopy {
                        buffer_offset: staging_offset as vk::DeviceSize,
                        buffer_row_length: desc.width,
                        buffer_image_height: desc.height_or_zero.max(1),
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: desc.aspect_mask,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: desc.layer_count_or_zero.max(1),
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: desc.width,
                            height: desc.height_or_zero.max(1),
                            depth: 1,
                        },
                    };

                    unsafe {
                        self.shared.context.device.cmd_copy_buffer_to_image(
                            cmd,
                            self.shared.staging_buffer.0,
                            image.0,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            slice::from_ref(&region),
                        )
                    };

                    emit_image_barrier(
                        ImageUsage::TRANSFER_WRITE,
                        all_usage,
                        image.0,
                        desc.aspect_mask,
                        &self.shared.context.device,
                        cmd,
                    );

                    let mut images = self.shared.images.lock().unwrap();
                    let resource = images.get_mut(handle.0).unwrap();
                    assert!(matches!(resource, StaticImageResource::Mapped));
                    *resource = StaticImageResource::Ready {
                        desc,
                        image,
                        image_view: self.resource_cache.get_image_view(&desc, image),
                    };
                }
            }

            self.frees.push(ResourceStagingFree {
                staging_offset,
                countdown: CommandBufferPool::COUNT as u32,
            });
        }
    }

    pub fn get_buffer_desc(&self, handle: StaticBufferHandle) -> Option<BufferDesc> {
        let buffers = self.shared.buffers.lock().unwrap();
        buffers
            .get(handle.0)
            .and_then(|r| match r {
                StaticBufferResource::Ready { desc, .. } => Some(desc),
                _ => None,
            })
            .copied()
    }

    pub fn get_buffer(&self, handle: StaticBufferHandle) -> Option<vk::Buffer> {
        let buffers = self.shared.buffers.lock().unwrap();
        buffers.get(handle.0).and_then(|r| match r {
            StaticBufferResource::Ready { buffer, .. } => Some(buffer.0),
            _ => None,
        })
    }

    pub fn get_image_desc(&self, handle: StaticImageHandle) -> Option<ImageDesc> {
        let images = self.shared.images.lock().unwrap();
        images
            .get(handle.0)
            .and_then(|r| match r {
                StaticImageResource::Ready { desc, .. } => Some(desc),
                _ => None,
            })
            .copied()
    }

    pub fn get_image(&self, handle: StaticImageHandle) -> Option<vk::Image> {
        let images = self.shared.images.lock().unwrap();
        images.get(handle.0).and_then(|r| match r {
            StaticImageResource::Ready { image, .. } => Some(image.0),
            _ => None,
        })
    }

    pub fn get_image_view(&self, handle: StaticImageHandle) -> Option<vk::ImageView> {
        let images = self.shared.images.lock().unwrap();
        images.get(handle.0).and_then(|r| match r {
            StaticImageResource::Ready { image_view, .. } => Some(image_view.0),
            _ => None,
        })
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        ui.text("loader buffers");
        ui.next_column();
        ui.text(format!("{}", self.shared.buffers.lock().unwrap().active_count()));
        ui.next_column();

        ui.text("loader images");
        ui.next_column();
        ui.text(format!("{}", self.shared.images.lock().unwrap().active_count()));
        ui.next_column();

        self.resource_cache.ui_stats_table_rows(ui, "loader");
    }
}

impl Drop for ResourceLoader {
    fn drop(&mut self) {
        if let Some(thread_join) = self.writer_thread_join.take() {
            self.shared.exit_signal.store(true, Ordering::Release);
            self.shared.write_added.notify_all();
            self.shared.heap_freed.notify_all();
            println!("async load stopping!");
            thread_join.join().unwrap();
        }
    }
}

pub struct ResourceWriter<'a> {
    shared: Arc<ResourceLoaderShared>,
    mapping: &'a mut [u8],
    next: usize,
    transfer: ResourceStagingTransfer,
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

impl<'a> ResourceWriter<'a> {
    pub fn write<T: AsBytes + ?Sized>(&mut self, pod: &T) {
        let bytes = pod.as_bytes();
        let start = self.next;
        let end = start + bytes.len();
        self.mapping[start..end].copy_from_slice(bytes);
        self.next += bytes.len();
    }

    pub fn written(&self) -> usize {
        self.next
    }

    pub fn write_zeros(&mut self, len: usize) {
        let start = self.next;
        let end = start + len;
        for dst in self.mapping[start..end].iter_mut() {
            *dst = 0;
        }
        self.next += len;
    }
}

impl<'a> Drop for ResourceWriter<'a> {
    fn drop(&mut self) {
        for dst in self.mapping[self.next..].iter_mut() {
            *dst = 0;
        }
        let mut transfers = self.shared.transfers.lock().unwrap();
        transfers.push_back(self.transfer);
    }
}

/*
   Want to define resource asynchronously (e.g. after loading file).
   So need to allocate GPU mem either:
   * On writer thread (when mapping)>
   * When doing transfer (potentially on async transfer thread?)

   Aim for late as possible, can share allocator later to optimise.
*/
pub struct ResourceAllocator {
    shared: Arc<ResourceLoaderShared>,
}

impl ResourceAllocator {
    fn map(&self, size: usize) -> Option<(u32, &mut [u8])> {
        let alignment = self
            .shared
            .context
            .physical_device_properties
            .limits
            .min_storage_buffer_offset_alignment as u32;
        let staging_offset = self.shared.allocate_staging(size as u32, alignment)?;
        let mapping = unsafe {
            slice::from_raw_parts_mut(
                self.shared.staging_mapping.0.offset(staging_offset as isize),
                size as usize,
            )
        };
        Some((staging_offset, mapping))
    }

    pub fn map_buffer(
        &self,
        handle: StaticBufferHandle,
        desc: &BufferDesc,
        all_usage: BufferUsage,
    ) -> Option<ResourceWriter<'_>> {
        {
            let mut buffers = self.shared.buffers.lock().unwrap();
            let resource = buffers.get_mut(handle.0).unwrap();
            assert!(matches!(resource, StaticBufferResource::Empty));
            *resource = StaticBufferResource::Mapped;
        }
        let (staging_offset, mapping) = self.map(desc.size)?;
        Some(ResourceWriter {
            shared: Arc::clone(&self.shared),
            mapping,
            next: 0,
            transfer: ResourceStagingTransfer {
                resource: ResourceStagingTransferResource::Buffer {
                    handle,
                    desc: *desc,
                    all_usage,
                },
                staging_offset,
            },
        })
    }

    pub fn map_image(
        &self,
        handle: StaticImageHandle,
        desc: &ImageDesc,
        all_usage: ImageUsage,
    ) -> Option<ResourceWriter<'_>> {
        {
            let mut images = self.shared.images.lock().unwrap();
            let resource = images.get_mut(handle.0).unwrap();
            assert!(matches!(resource, StaticImageResource::Empty));
            *resource = StaticImageResource::Mapped;
        }
        let (staging_offset, mapping) = self.map(desc.staging_size())?;
        Some(ResourceWriter {
            shared: Arc::clone(&self.shared),
            mapping,
            next: 0,
            transfer: ResourceStagingTransfer {
                resource: ResourceStagingTransferResource::Image {
                    handle,
                    desc: *desc,
                    all_usage,
                },
                staging_offset,
            },
        })
    }
}
