use crate::command_buffer::CommandBufferPool;
use crate::context::Context;
use arrayvec::{Array, ArrayVec};
use imgui::{ProgressBar, Ui};
use spark::{vk, Builder};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn align_up(x: u32, alignment: u32) -> u32 {
    (x + alignment - 1) & !(alignment - 1)
}

#[repr(align(4))]
struct AlignedArray<A: Array>(A);

unsafe impl<A: Array> Array for AlignedArray<A> {
    type Item = A::Item;
    type Index = A::Index;
    const CAPACITY: usize = A::CAPACITY;
    fn as_slice(&self) -> &[Self::Item] {
        self.0.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self.0.as_mut_slice()
    }
}

struct UniformDataPool {
    context: Arc<Context>,
    min_alignment: u32,
    atom_size: u32,
    size_per_frame: u32,
    mem: vk::DeviceMemory,
    mapping: *mut u8,
    buffers: [vk::Buffer; Self::COUNT],
    buffer_index: usize,
    next_offset: AtomicUsize,
    last_usage: u32,
}

impl UniformDataPool {
    const COUNT: usize = CommandBufferPool::COUNT;

    pub fn new(context: &Arc<Context>, size_per_frame: u32) -> Self {
        let min_alignment = context
            .physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment as u32;
        let atom_size = context.physical_device_properties.limits.non_coherent_atom_size as u32;
        let size_per_frame = size_per_frame.min(context.physical_device_properties.limits.max_uniform_buffer_range);

        let mut memory_type_filter = 0xffff_ffff;
        let buffers: [vk::Buffer; Self::COUNT] = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: vk::DeviceSize::from(size_per_frame),
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                ..Default::default()
            };
            let mut buffers = ArrayVec::new();
            for _i in 0..Self::COUNT {
                let buffer = unsafe { context.device.create_buffer(&buffer_create_info, None) }.unwrap();
                let mem_req = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                assert!(mem_req.size == buffer_create_info.size);
                buffers.push(buffer);
                memory_type_filter &= mem_req.memory_type_bits;
            }
            buffers.into_inner().unwrap()
        };

        let mem = {
            let memory_type_index = context
                .get_memory_type_index(memory_type_filter, vk::MemoryPropertyFlags::HOST_VISIBLE)
                .unwrap();
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: (Self::COUNT * (size_per_frame as usize)) as vk::DeviceSize,
                memory_type_index,
                ..Default::default()
            };
            unsafe { context.device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };

        for (i, buffer) in buffers.iter().enumerate() {
            unsafe {
                context
                    .device
                    .bind_buffer_memory(*buffer, mem, (i * (size_per_frame as usize)) as vk::DeviceSize)
            }
            .unwrap();
        }

        let mapping = unsafe { context.device.map_memory(mem, 0, vk::WHOLE_SIZE, Default::default()) }.unwrap();

        Self {
            context: Arc::clone(&context),
            min_alignment,
            atom_size,
            size_per_frame,
            mem,
            mapping: mapping as *mut _,
            buffers,
            buffer_index: 0,
            next_offset: AtomicUsize::new(0),
            last_usage: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.buffer_index = (self.buffer_index + 1) % Self::COUNT;
        self.next_offset = AtomicUsize::new(0);
    }

    pub fn end_frame(&mut self) {
        self.last_usage = self.next_offset.load(Ordering::Acquire) as u32;
        if self.last_usage == 0 {
            return;
        }

        let mapped_ranges = [vk::MappedMemoryRange {
            memory: Some(self.mem),
            offset: (self.buffer_index * (self.size_per_frame as usize)) as vk::DeviceSize,
            size: vk::DeviceSize::from(align_up(self.last_usage, self.atom_size)),
            ..Default::default()
        }];
        unsafe { self.context.device.flush_mapped_memory_ranges(&mapped_ranges) }.unwrap();
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffers[self.buffer_index]
    }

    pub fn alloc(&self, size: u32) -> Option<(&mut [u8], u32)> {
        let aligned_size = align_up(size, self.min_alignment);

        let base = self.next_offset.fetch_add(aligned_size as usize, Ordering::SeqCst) as u32;
        let end = base + aligned_size;

        if end <= self.size_per_frame {
            Some((
                unsafe {
                    slice::from_raw_parts_mut(
                        self.mapping
                            .add(self.buffer_index * (self.size_per_frame as usize))
                            .add(base as usize),
                        size as usize,
                    )
                },
                base,
            ))
        } else {
            None
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        ui.text("uniform data");
        ui.next_column();
        ProgressBar::new((self.last_usage as f32) / (self.size_per_frame as f32)).build(ui);
        ui.next_column();
    }
}

impl Drop for UniformDataPool {
    fn drop(&mut self) {
        unsafe {
            for buffer in self.buffers.iter() {
                self.context.device.destroy_buffer(Some(*buffer), None);
            }
            self.context.device.unmap_memory(self.mem);
            self.context.device.free_memory(Some(self.mem), None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DescriptorSetLayoutBinding {
    SampledImage { sampler: vk::Sampler },
    StorageImage,
    UniformData { size: u32 },
    StorageBuffer,
    AccelerationStructure,
}

pub enum DescriptorSetBindingData<'a> {
    SampledImage { image_view: vk::ImageView },
    StorageImage { image_view: vk::ImageView },
    UniformData { size: u32, writer: &'a dyn Fn(&mut [u8]) },
    StorageBuffer { buffer: vk::Buffer },
    AccelerationStructure { accel: vk::AccelerationStructureKHR },
}

pub struct DescriptorSetLayoutCache {
    context: Arc<Context>,
    use_inline_uniform_block: bool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layouts: Vec<vk::PipelineLayout>,
}

impl DescriptorSetLayoutCache {
    pub fn new(context: &Arc<Context>) -> Self {
        let use_inline_uniform_block = context.device.extensions.supports_ext_inline_uniform_block();
        Self {
            context: Arc::clone(context),
            use_inline_uniform_block,
            descriptor_set_layouts: Vec::new(),
            pipeline_layouts: Vec::new(),
        }
    }

    pub fn create_descriptor_set_layout(&mut self, bindings: &[DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let mut bindings_vk = Vec::new();
        for (i, binding) in bindings.iter().enumerate() {
            match binding {
                DescriptorSetLayoutBinding::SampledImage { sampler } => {
                    bindings_vk.push(vk::DescriptorSetLayoutBinding {
                        binding: i as u32,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        p_immutable_samplers: sampler,
                    });
                }
                DescriptorSetLayoutBinding::StorageImage => {
                    bindings_vk.push(vk::DescriptorSetLayoutBinding {
                        binding: i as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        ..Default::default()
                    });
                }
                DescriptorSetLayoutBinding::UniformData { size } => {
                    if self.use_inline_uniform_block {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                            descriptor_count: *size,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    } else {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    }
                }
                DescriptorSetLayoutBinding::StorageBuffer => bindings_vk.push(vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                }),
                DescriptorSetLayoutBinding::AccelerationStructure => bindings_vk.push(vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                }),
            }
        }
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().p_bindings(&bindings_vk);
        let descriptor_set_layout =
            unsafe { self.context.device.create_descriptor_set_layout(&create_info, None) }.unwrap();
        self.descriptor_set_layouts.push(descriptor_set_layout);
        descriptor_set_layout
    }

    pub fn create_pipeline_layout(&mut self, descriptor_set_layout: vk::DescriptorSetLayout) -> vk::PipelineLayout {
        let create_info =
            vk::PipelineLayoutCreateInfo::builder().p_set_layouts(slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe { self.context.device.create_pipeline_layout(&create_info, None) }.unwrap();
        self.pipeline_layouts.push(pipeline_layout);
        pipeline_layout
    }
}

impl Drop for DescriptorSetLayoutCache {
    fn drop(&mut self) {
        let device = &self.context.device;
        for pipeline_layout in self.pipeline_layouts.iter() {
            unsafe { device.destroy_pipeline_layout(Some(*pipeline_layout), None) };
        }
        for descriptor_set_layout in self.descriptor_set_layouts.iter() {
            unsafe { device.destroy_descriptor_set_layout(Some(*descriptor_set_layout), None) };
        }
    }
}

pub struct DescriptorPool {
    context: Arc<Context>,
    pools: [vk::DescriptorPool; Self::COUNT],
    pool_index: usize,
    uniform_data_pool: Option<UniformDataPool>,
}

impl DescriptorPool {
    const COUNT: usize = CommandBufferPool::COUNT;

    // per frame maximums
    const MAX_DESCRIPTORS_PER_FRAME: u32 = 512;
    const MAX_SETS_PER_FRAME: u32 = 512;
    const MAX_UNIFORM_DATA_PER_FRAME: u32 = 64 * 1024;

    const MAX_UNIFORM_DATA_PER_SET: usize = 2 * 1024;
    const MAX_DESCRIPTORS_PER_SET: usize = 16;

    pub fn new(context: &Arc<Context>) -> Self {
        let use_inline_uniform_block = context.device.extensions.supports_ext_inline_uniform_block();
        if use_inline_uniform_block {
            println!("using inline uniform block for uniform data");
        }

        let pools = {
            let mut descriptor_pool_sizes = ArrayVec::<[_; 5]>::new();
            descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
            });
            descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
            });
            descriptor_pool_sizes.push(if use_inline_uniform_block {
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                    descriptor_count: Self::MAX_UNIFORM_DATA_PER_FRAME,
                }
            } else {
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
                }
            });
            descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
            });
            if context.device.extensions.supports_khr_acceleration_structure() {
                descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
                });
            }

            let mut inline_uniform_block_create_info = vk::DescriptorPoolInlineUniformBlockCreateInfoEXT::builder()
                .max_inline_uniform_block_bindings(if use_inline_uniform_block {
                    Self::MAX_DESCRIPTORS_PER_FRAME
                } else {
                    0
                });

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(Self::MAX_SETS_PER_FRAME)
                .p_pool_sizes(&descriptor_pool_sizes)
                .insert_next(&mut inline_uniform_block_create_info);

            let mut pools = ArrayVec::new();
            for _i in 0..Self::COUNT {
                pools.push(
                    unsafe {
                        context
                            .device
                            .create_descriptor_pool(&descriptor_pool_create_info, None)
                    }
                    .unwrap(),
                );
            }
            pools.into_inner().unwrap()
        };
        Self {
            context: Arc::clone(&context),
            pools,
            pool_index: 0,
            uniform_data_pool: if use_inline_uniform_block {
                None
            } else {
                Some(UniformDataPool::new(context, Self::MAX_UNIFORM_DATA_PER_FRAME))
            },
        }
    }

    pub fn begin_frame(&mut self) {
        unsafe {
            self.context
                .device
                .reset_descriptor_pool(self.pools[self.pool_index], vk::DescriptorPoolResetFlags::empty())
        }
        .unwrap();
        if let Some(uniform_data_pool) = self.uniform_data_pool.as_mut() {
            uniform_data_pool.begin_frame();
        }
    }

    pub fn create_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
        data: &[DescriptorSetBindingData],
    ) -> vk::DescriptorSet {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pools[self.pool_index])
            .p_set_layouts(slice::from_ref(&layout));
        let descriptor_set = unsafe {
            self.context
                .device
                .allocate_descriptor_sets_single(&descriptor_set_allocate_info)
        }
        .unwrap();

        let mut buffer_info = ArrayVec::<[_; Self::MAX_DESCRIPTORS_PER_SET]>::new();
        let mut image_info = ArrayVec::<[_; Self::MAX_DESCRIPTORS_PER_SET]>::new();
        let mut writes = ArrayVec::<[_; Self::MAX_DESCRIPTORS_PER_SET]>::new();
        let mut inline_writes = ArrayVec::<[_; Self::MAX_DESCRIPTORS_PER_SET]>::new();
        let mut inline_uniform_data = ArrayVec::<AlignedArray<[u8; Self::MAX_UNIFORM_DATA_PER_SET]>>::new();
        let mut acceleration_structure_writes = ArrayVec::<[_; Self::MAX_DESCRIPTORS_PER_SET]>::new();

        for (i, data) in data.iter().enumerate() {
            match data {
                DescriptorSetBindingData::SampledImage { image_view } => {
                    image_info.push(vk::DescriptorImageInfo {
                        sampler: None,
                        image_view: Some(*image_view),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    });

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: image_info.last().unwrap(),
                        ..Default::default()
                    });
                }
                DescriptorSetBindingData::StorageImage { image_view } => {
                    image_info.push(vk::DescriptorImageInfo {
                        sampler: None,
                        image_view: Some(*image_view),
                        image_layout: vk::ImageLayout::GENERAL,
                    });

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: image_info.last().unwrap(),
                        ..Default::default()
                    });
                }
                DescriptorSetBindingData::UniformData { size, writer } => {
                    if let Some(uniform_data_pool) = self.uniform_data_pool.as_ref() {
                        // write uniform data into buffer
                        let size = *size;
                        let (addr, offset) = uniform_data_pool.alloc(size).unwrap();
                        writer(addr);

                        buffer_info.push(vk::DescriptorBufferInfo {
                            buffer: Some(uniform_data_pool.get_buffer()),
                            offset: vk::DeviceSize::from(offset),
                            range: vk::DeviceSize::from(size),
                        });

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: Some(descriptor_set),
                            dst_binding: i as u32,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            p_buffer_info: buffer_info.last().unwrap(),
                            ..Default::default()
                        });
                    } else {
                        // write uniform data to the stack
                        let size = *size as usize;
                        let data_ptr = inline_uniform_data.as_mut_ptr().wrapping_add(inline_uniform_data.len());
                        if inline_uniform_data.len() + size > inline_uniform_data.capacity() {
                            panic!("not enough space to write inline uniform data");
                        }
                        unsafe {
                            inline_uniform_data.set_len(inline_uniform_data.len() + size);
                        }
                        writer(unsafe { slice::from_raw_parts_mut(data_ptr, size) });

                        inline_writes.push(vk::WriteDescriptorSetInlineUniformBlockEXT {
                            data_size: size as u32,
                            p_data: data_ptr as *const _,
                            ..Default::default()
                        });

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: Some(descriptor_set),
                            dst_binding: i as u32,
                            descriptor_count: size as u32,
                            descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                            p_next: inline_writes.last().unwrap() as *const _ as *const _,
                            ..Default::default()
                        });
                    }
                }
                DescriptorSetBindingData::StorageBuffer { buffer } => {
                    buffer_info.push(vk::DescriptorBufferInfo {
                        buffer: Some(*buffer),
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    });

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: buffer_info.last().unwrap(),
                        ..Default::default()
                    });
                }
                DescriptorSetBindingData::AccelerationStructure { accel } => {
                    acceleration_structure_writes.push(vk::WriteDescriptorSetAccelerationStructureKHR {
                        acceleration_structure_count: 1,
                        p_acceleration_structures: accel,
                        ..Default::default()
                    });

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        p_next: acceleration_structure_writes.last().unwrap() as *const _ as *const _,
                        ..Default::default()
                    });
                }
            }
        }

        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };

        descriptor_set
    }

    pub fn end_command_buffer(&mut self) {
        self.pool_index = (self.pool_index + 1) % Self::COUNT;
        if let Some(uniform_data_pool) = self.uniform_data_pool.as_mut() {
            uniform_data_pool.end_frame();
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        if let Some(uniform_data_pool) = self.uniform_data_pool.as_ref() {
            uniform_data_pool.ui_stats_table_rows(ui);
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        let device = &self.context.device;
        for pool in self.pools.iter() {
            unsafe {
                device.destroy_descriptor_pool(Some(*pool), None);
            }
        }
    }
}
