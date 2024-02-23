use crate::{command_buffer::CommandBufferPool, context::*};
use arrayvec::ArrayVec;
use spark::{vk, Builder};
use std::{
    any::TypeId,
    cell::{Cell, RefCell},
    collections::HashMap,
    slice,
};

fn align_up(x: u32, alignment: u32) -> u32 {
    (x + alignment - 1) & !(alignment - 1)
}

pub(crate) struct StagingBuffer {
    context: SharedContext,
    min_alignment: u32,
    atom_size: u32,
    size_per_frame: u32,
    mem: vk::DeviceMemory,
    mapping: *mut u8,
    buffers: [vk::Buffer; Self::COUNT],
    buffer_index: usize,
    next_offset: Cell<u32>,
    last_usage: u32,
}

impl StagingBuffer {
    const COUNT: usize = CommandBufferPool::COUNT;

    pub fn new(
        context: &SharedContext,
        size_per_frame: u32,
        min_alignment: u32,
        usage_flags: vk::BufferUsageFlags,
    ) -> Self {
        let atom_size = context.physical_device_properties.limits.non_coherent_atom_size as u32;

        let mut memory_type_filter = 0xffff_ffff;
        let buffers: [vk::Buffer; Self::COUNT] = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: vk::DeviceSize::from(size_per_frame),
                usage: usage_flags,
                ..Default::default()
            };
            let mut buffers = ArrayVec::new();
            for _i in 0..Self::COUNT {
                let buffer = unsafe { context.device.create_buffer(&buffer_create_info, None) }.unwrap();
                let mem_req = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                assert_eq!(mem_req.size, buffer_create_info.size);
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
            context: SharedContext::clone(context),
            min_alignment,
            atom_size,
            size_per_frame,
            mem,
            mapping: mapping as *mut _,
            buffers,
            buffer_index: 0,
            next_offset: Cell::new(0),
            last_usage: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.buffer_index = (self.buffer_index + 1) % Self::COUNT;
        self.next_offset = Cell::new(0);
    }

    pub fn end_frame(&mut self) {
        self.last_usage = self.next_offset.get();
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

    pub fn alloc(&self, size: u32, align: u32) -> Option<(&mut [u8], u32)> {
        let base = self.next_offset.get();
        let aligned_base = align_up(base, self.min_alignment.max(align));
        let end = aligned_base + size;
        self.next_offset.set(end);

        if end <= self.size_per_frame {
            Some((
                unsafe {
                    slice::from_raw_parts_mut(
                        self.mapping
                            .add(self.buffer_index * (self.size_per_frame as usize))
                            .add(aligned_base as usize),
                        size as usize,
                    )
                },
                base,
            ))
        } else {
            None
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &mut egui::Ui, title: &str) {
        ui.label(title);
        ui.add(egui::ProgressBar::new(
            (self.last_usage as f32) / (self.size_per_frame as f32),
        ));
        ui.end_row();
    }
}

impl Drop for StagingBuffer {
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

#[derive(Clone, Copy)]
pub struct DescriptorSet {
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
}

#[derive(Debug, Clone, Copy)]
pub enum DescriptorSetLayoutBinding {
    Sampler,
    SampledImage,
    CombinedImageSampler,
    StorageImage { count: u32 },
    UniformData { size: u32 },
    StorageBuffer,
    AccelerationStructure,
}

pub enum DescriptorSetBindingData<'a> {
    Sampler {
        sampler: vk::Sampler,
    },
    SampledImage {
        image_view: vk::ImageView,
    },
    CombinedImageSampler {
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    },
    StorageImage {
        image_views: &'a [vk::ImageView],
    },
    UniformData {
        size: u32,
        align: u32,
        writer: &'a dyn Fn(&mut [u8]),
    },
    StorageBuffer {
        buffer: vk::Buffer,
    },
    AccelerationStructure {
        accel: vk::AccelerationStructureKHR,
    },
}

struct DescriptorSetLayoutCache {
    context: SharedContext,
    use_inline_uniform_block: bool,
    layouts: HashMap<TypeId, vk::DescriptorSetLayout>,
}

impl DescriptorSetLayoutCache {
    fn new(context: &SharedContext, use_inline_uniform_block: bool) -> Self {
        Self {
            context: SharedContext::clone(context),
            use_inline_uniform_block,
            layouts: HashMap::new(),
        }
    }

    pub fn get_layout(&mut self, key: TypeId, bindings: &[DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let device = &self.context.device;
        let use_inline_uniform_block = self.use_inline_uniform_block;
        *self.layouts.entry(key).or_insert_with(|| {
            let mut bindings_vk = Vec::new();
            for (i, binding) in bindings.iter().enumerate() {
                match binding {
                    DescriptorSetLayoutBinding::Sampler => {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    }
                    DescriptorSetLayoutBinding::SampledImage => {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    }
                    DescriptorSetLayoutBinding::CombinedImageSampler => {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    }
                    DescriptorSetLayoutBinding::StorageImage { count } => {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: *count,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        });
                    }
                    DescriptorSetLayoutBinding::UniformData { size } => {
                        if use_inline_uniform_block {
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
                    DescriptorSetLayoutBinding::AccelerationStructure => {
                        bindings_vk.push(vk::DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::ALL,
                            ..Default::default()
                        })
                    }
                }
            }
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().p_bindings(&bindings_vk);
            unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap()
        })
    }
}

impl Drop for DescriptorSetLayoutCache {
    fn drop(&mut self) {
        let device = &self.context.device;
        for (_, layout) in self.layouts.drain() {
            unsafe { device.destroy_descriptor_set_layout(Some(layout), None) };
        }
    }
}

pub struct DescriptorPool {
    context: SharedContext,
    layout_cache: RefCell<DescriptorSetLayoutCache>,
    pools: [vk::DescriptorPool; Self::COUNT],
    pool_index: usize,
    uniform_data_pool: Option<StagingBuffer>,
}

impl DescriptorPool {
    const COUNT: usize = CommandBufferPool::COUNT;

    // per frame maximums
    const MAX_DESCRIPTORS_PER_FRAME: u32 = 64 * 1024;
    const MAX_SETS_PER_FRAME: u32 = 64 * 1024;
    const MAX_UNIFORM_DATA_PER_FRAME: u32 = 1 * 1024 * 1024;

    const MAX_UNIFORM_DATA_PER_SET: usize = 2 * 1024;
    const MAX_DESCRIPTORS_PER_SET: usize = 16;

    pub fn new(context: &SharedContext) -> Self {
        let use_inline_uniform_block = context
            .physical_device_features
            .inline_uniform_block
            .inline_uniform_block
            .as_bool();
        if use_inline_uniform_block {
            println!("using inline uniform block for uniform data");
        }

        let layout_cache = DescriptorSetLayoutCache::new(context, use_inline_uniform_block);

        let pools = {
            let mut descriptor_pool_sizes = Vec::new();
            descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
            });
            descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: Self::MAX_DESCRIPTORS_PER_FRAME,
            });
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
            if context
                .physical_device_features
                .acceleration_structure
                .acceleration_structure
                .as_bool()
            {
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
            context: SharedContext::clone(context),
            layout_cache: RefCell::new(layout_cache),
            pools,
            pool_index: 0,
            uniform_data_pool: if use_inline_uniform_block {
                None
            } else {
                let size_per_frame = Self::MAX_UNIFORM_DATA_PER_FRAME
                    .min(context.physical_device_properties.limits.max_uniform_buffer_range);
                let min_alignment = context
                    .physical_device_properties
                    .limits
                    .min_uniform_buffer_offset_alignment as u32;
                Some(StagingBuffer::new(
                    context,
                    size_per_frame,
                    min_alignment,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ))
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

    pub fn get_descriptor_set_layout(
        &self,
        key: TypeId,
        bindings: &[DescriptorSetLayoutBinding],
    ) -> vk::DescriptorSetLayout {
        self.layout_cache.borrow_mut().get_layout(key, bindings)
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

        let mut buffer_info = ArrayVec::<_, { Self::MAX_DESCRIPTORS_PER_SET }>::new();
        let mut image_info = ArrayVec::<_, { Self::MAX_DESCRIPTORS_PER_SET }>::new();
        let mut writes = ArrayVec::<_, { Self::MAX_DESCRIPTORS_PER_SET }>::new();
        let mut inline_writes = ArrayVec::<_, { Self::MAX_DESCRIPTORS_PER_SET }>::new();
        let mut inline_uniform_data = ArrayVec::<u8, { Self::MAX_UNIFORM_DATA_PER_SET }>::new();
        let mut acceleration_structure_writes = ArrayVec::<_, { Self::MAX_DESCRIPTORS_PER_SET }>::new();

        for (i, data) in data.iter().enumerate() {
            match data {
                DescriptorSetBindingData::Sampler { sampler } => {
                    image_info.push(vk::DescriptorImageInfo {
                        sampler: Some(*sampler),
                        image_view: None,
                        image_layout: vk::ImageLayout::UNDEFINED,
                    });

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        p_image_info: image_info.last().unwrap(),
                        ..Default::default()
                    });
                }
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
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: image_info.last().unwrap(),
                        ..Default::default()
                    });
                }
                DescriptorSetBindingData::CombinedImageSampler { image_view, sampler } => {
                    image_info.push(vk::DescriptorImageInfo {
                        sampler: Some(*sampler),
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
                DescriptorSetBindingData::StorageImage { image_views } => {
                    let offset = image_info.len();
                    for image_view in image_views.iter() {
                        image_info.push(vk::DescriptorImageInfo {
                            sampler: None,
                            image_view: Some(*image_view),
                            image_layout: vk::ImageLayout::GENERAL,
                        });
                    }

                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: image_views.len() as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: image_info.get(offset).unwrap(),
                        ..Default::default()
                    });
                }
                DescriptorSetBindingData::UniformData { size, align, writer } => {
                    let (align, size) = (*align, *size);
                    if let Some(uniform_data_pool) = self.uniform_data_pool.as_ref() {
                        // write uniform data into buffer
                        let (addr, offset) = uniform_data_pool.alloc(size, align).unwrap();
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
                        let start_offset = inline_uniform_data.len();
                        let end_offset = start_offset + ((align + size) as usize);
                        if end_offset > inline_uniform_data.capacity() {
                            panic!("not enough space to write inline uniform data");
                        }
                        let block = unsafe {
                            inline_uniform_data.set_len(end_offset);
                            let start_ptr = inline_uniform_data.as_mut_ptr().add(start_offset);
                            let align_offset = start_ptr.align_offset(align as usize);
                            slice::from_raw_parts_mut(start_ptr.add(align_offset), size as usize)
                        };
                        writer(block);

                        inline_writes.push(vk::WriteDescriptorSetInlineUniformBlockEXT {
                            data_size: size,
                            p_data: block.as_ptr() as *const _,
                            ..Default::default()
                        });

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: Some(descriptor_set),
                            dst_binding: i as u32,
                            descriptor_count: size,
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

    pub fn end_frame(&mut self) {
        self.pool_index = (self.pool_index + 1) % Self::COUNT;
        if let Some(uniform_data_pool) = self.uniform_data_pool.as_mut() {
            uniform_data_pool.end_frame();
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &mut egui::Ui) {
        if let Some(uniform_data_pool) = self.uniform_data_pool.as_ref() {
            uniform_data_pool.ui_stats_table_rows(ui, "uniform data");
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
