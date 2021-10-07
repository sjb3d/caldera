use crate::prelude::*;
use bytemuck::Contiguous;
use imgui::Ui;
use slotmap::{new_key_type, SlotMap};
use spark::{vk, Builder, Device};
use std::{
    slice,
    sync::{Arc, Mutex},
};

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
        accel: Option<vk::AccelerationStructureKHR>,
        bindless_id: Option<BindlessId>,
        current_usage: BufferUsage,
        all_usage_check: BufferUsage,
    },
}

impl BufferResource {
    pub fn desc(&self) -> &BufferDesc {
        match self {
            BufferResource::Described { desc, .. } => desc,
            BufferResource::Active { desc, .. } => desc,
        }
    }

    pub fn alloc(&self) -> Option<Alloc> {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { alloc, .. } => *alloc,
        }
    }

    pub fn buffer(&self) -> UniqueBuffer {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { buffer, .. } => *buffer,
        }
    }

    pub fn accel(&self) -> Option<vk::AccelerationStructureKHR> {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { accel, .. } => *accel,
        }
    }

    pub fn bindless_id(&self) -> Option<BindlessId> {
        match self {
            BufferResource::Described { .. } => panic!("buffer is only described"),
            BufferResource::Active { bindless_id, .. } => *bindless_id,
        }
    }

    pub fn declare_usage(&mut self, usage: BufferUsage) {
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

    pub fn transition_usage(&mut self, new_usage: BufferUsage, device: &Device, cmd: vk::CommandBuffer) {
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
        bindless_id: Option<BindlessId>,
        current_usage: ImageUsage,
        all_usage_check: ImageUsage,
    },
}

impl ImageResource {
    pub fn desc(&self) -> &ImageDesc {
        match self {
            ImageResource::Described { desc, .. } => desc,
            ImageResource::Active { desc, .. } => desc,
        }
    }

    pub fn image(&self) -> UniqueImage {
        match self {
            ImageResource::Described { .. } => panic!("image is only described"),
            ImageResource::Active { image, .. } => *image,
        }
    }

    pub fn image_view(&self) -> UniqueImageView {
        match self {
            ImageResource::Described { .. } => panic!("image is only described"),
            ImageResource::Active { image_view, .. } => *image_view,
        }
    }

    pub fn bindless_id(&self) -> Option<BindlessId> {
        match self {
            ImageResource::Described { .. } => panic!("image is only described"),
            ImageResource::Active { bindless_id, .. } => *bindless_id,
        }
    }

    pub fn declare_usage(&mut self, usage: ImageUsage) {
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

    pub fn transition_usage(&mut self, new_usage: ImageUsage, device: &Device, cmd: vk::CommandBuffer) {
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

    pub fn force_usage(&mut self, new_usage: ImageUsage) {
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

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Contiguous)]
pub enum BindlessClass {
    StorageBuffer,
    SampledImage2D,
}

impl BindlessClass {
    fn new_buffer(all_usage: BufferUsage) -> Option<Self> {
        if all_usage.as_flags().contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            Some(Self::StorageBuffer)
        } else {
            None
        }
    }

    fn new_image(desc: &ImageDesc, all_usage: ImageUsage) -> Option<Self> {
        if desc.image_view_type() == vk::ImageViewType::N2D
            && all_usage.as_flags().contains(vk::ImageUsageFlags::SAMPLED)
        {
            Some(Self::SampledImage2D)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub struct BindlessId {
    pub class: BindlessClass,
    pub index: u16,
}

#[derive(Clone)]
struct BindlessIndexSet {
    next: u32,
    limit: u32,
}

impl BindlessIndexSet {
    fn new(limit: u32) -> Self {
        Self { next: 0, limit }
    }

    fn allocate(&mut self) -> Option<u32> {
        if self.next < self.limit {
            let index = self.next;
            self.next += 1;
            Some(index)
        } else {
            None
        }
    }
}

pub(crate) struct Bindless {
    context: SharedContext,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    indices: [BindlessIndexSet; Self::CLASS_COUNT],
}

impl Bindless {
    const CLASS_COUNT: usize = (1 + BindlessClass::MAX_VALUE) as usize;

    const MAX_STORAGE_BUFFER: u32 = 16 * 1024;
    const MAX_SAMPLED_IMAGE_2D: u32 = 1024;

    pub fn new(context: &SharedContext) -> Self {
        let descriptor_set_layout = {
            let bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: BindlessClass::StorageBuffer.into_integer() as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: Self::MAX_STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: BindlessClass::SampledImage2D.into_integer() as u32,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: Self::MAX_SAMPLED_IMAGE_2D,
                    stage_flags: vk::ShaderStageFlags::ALL,
                    ..Default::default()
                },
            ];
            let binding_flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
                Self::CLASS_COUNT];
            let mut binding_flags_create_info =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().p_binding_flags(&binding_flags);
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .p_bindings(&bindings)
                .insert_next(&mut binding_flags_create_info);
            unsafe { context.device.create_descriptor_set_layout(&create_info, None) }.unwrap()
        };
        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: Self::MAX_STORAGE_BUFFER,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: Self::MAX_SAMPLED_IMAGE_2D,
                },
            ];
            let create_info = vk::DescriptorPoolCreateInfo::builder()
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                .max_sets(1)
                .p_pool_sizes(&pool_sizes);
            unsafe { context.device.create_descriptor_pool(&create_info, None) }.unwrap()
        };
        let descriptor_set = {
            let allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .p_set_layouts(slice::from_ref(&descriptor_set_layout));
            unsafe { context.device.allocate_descriptor_sets_single(&allocate_info) }.unwrap()
        };
        Self {
            context: SharedContext::clone(context),
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            indices: [
                BindlessIndexSet::new(Self::MAX_STORAGE_BUFFER),
                BindlessIndexSet::new(Self::MAX_SAMPLED_IMAGE_2D),
            ],
        }
    }

    pub fn add_buffer(&mut self, buffer: vk::Buffer, all_usage: BufferUsage) -> Option<BindlessId> {
        let class = BindlessClass::new_buffer(all_usage)?;
        let index = self.indices[class.into_integer() as usize].allocate()?;
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: Some(buffer),
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(class.into_integer() as u32)
            .dst_array_element(index)
            .p_buffer_info(slice::from_ref(&buffer_info))
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER);
        unsafe { self.context.device.update_descriptor_sets(slice::from_ref(&write), &[]) };
        Some(BindlessId {
            class,
            index: index as u16,
        })
    }

    pub fn add_image(
        &mut self,
        desc: &ImageDesc,
        image_view: vk::ImageView,
        all_usage: ImageUsage,
    ) -> Option<BindlessId> {
        let class = BindlessClass::new_image(desc, all_usage)?;
        let index = self.indices[class.into_integer() as usize].allocate()?;
        let image_info = vk::DescriptorImageInfo {
            sampler: None,
            image_view: Some(image_view),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(class.into_integer() as u32)
            .dst_array_element(index)
            .p_image_info(slice::from_ref(&image_info))
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE);
        unsafe { self.context.device.update_descriptor_sets(slice::from_ref(&write), &[]) };
        Some(BindlessId {
            class,
            index: index as u16,
        })
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        for class_index in BindlessClass::MIN_VALUE..=BindlessClass::MAX_VALUE {
            let class = BindlessClass::from_integer(class_index).unwrap();
            ui.text(match class {
                BindlessClass::StorageBuffer => "bindless buffers",
                BindlessClass::SampledImage2D => "bindless sampled2d",
            });
            ui.next_column();
            ui.text(format!("{}", self.indices[class_index as usize].next));
            ui.next_column();
        }
    }
}

impl Drop for Bindless {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe {
            device.destroy_descriptor_pool(Some(self.descriptor_pool), None);
            device.destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}

pub(crate) type SharedResources = Arc<Mutex<Resources>>;

pub(crate) struct Resources {
    global_allocator: Allocator,
    resource_cache: ResourceCache,
    buffers: SlotMap<BufferId, BufferResource>,
    images: SlotMap<ImageId, ImageResource>,
    bindless: Option<Bindless>,
}

impl Resources {
    pub fn new(context: &SharedContext, global_chunk_size: u32) -> SharedResources {
        Arc::new(Mutex::new(Self {
            global_allocator: Allocator::new(context, global_chunk_size),
            resource_cache: ResourceCache::new(context),
            buffers: SlotMap::with_key(),
            images: SlotMap::with_key(),
            bindless: if context.enable_bindless {
                Some(Bindless::new(context))
            } else {
                None
            },
        }))
    }

    pub fn bindless_descriptor_set(&self) -> vk::DescriptorSet {
        self.bindless.as_ref().unwrap().descriptor_set
    }

    pub fn bindless_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.bindless.as_ref().unwrap().descriptor_set_layout
    }

    pub fn describe_buffer(&mut self, desc: &BufferDesc) -> BufferId {
        self.buffers.insert(BufferResource::Described {
            desc: *desc,
            all_usage: BufferUsage::empty(),
        })
    }

    pub fn import_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        buffer: UniqueBuffer,
        current_usage: BufferUsage,
    ) -> BufferId {
        self.buffers.insert(BufferResource::Active {
            desc: *desc,
            alloc: None,
            buffer,
            accel: None,
            bindless_id: None,
            current_usage,
            all_usage_check: all_usage,
        })
    }

    pub fn create_buffer(
        &mut self,
        desc: &BufferDesc,
        all_usage: BufferUsage,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> BufferId {
        let all_usage_flags = all_usage.as_flags();
        let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
        let alloc = self.global_allocator.allocate(&info.mem_req, memory_property_flags);
        let buffer = self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags);
        let accel = self.resource_cache.get_buffer_accel(desc, buffer, all_usage);
        let bindless_id = self
            .bindless
            .as_mut()
            .and_then(|bindless| bindless.add_buffer(buffer.0, all_usage));
        self.buffers.insert(BufferResource::Active {
            desc: *desc,
            alloc: Some(alloc),
            buffer,
            accel,
            bindless_id,
            current_usage: BufferUsage::empty(),
            all_usage_check: all_usage,
        })
    }

    pub fn buffer_resource(&self, id: BufferId) -> &BufferResource {
        self.buffers.get(id).unwrap()
    }

    pub fn buffer_resource_mut(&mut self, id: BufferId) -> &mut BufferResource {
        self.buffers.get_mut(id).unwrap()
    }

    pub fn allocate_temporary_buffer(&mut self, id: BufferId, allocator: &mut Allocator) {
        let buffer_resource = self.buffers.get_mut(id).unwrap();
        *buffer_resource = match buffer_resource {
            BufferResource::Described { desc, all_usage } => {
                let all_usage_flags = all_usage.as_flags();
                let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                let info = self.resource_cache.get_buffer_info(desc, all_usage_flags);
                let alloc = allocator.allocate(&info.mem_req, memory_property_flags);
                let buffer = self.resource_cache.get_buffer(desc, &info, &alloc, all_usage_flags);
                let accel = self.resource_cache.get_buffer_accel(desc, buffer, *all_usage);
                BufferResource::Active {
                    desc: *desc,
                    alloc: Some(alloc),
                    buffer,
                    accel,
                    bindless_id: None,
                    current_usage: BufferUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("buffer is not temporary"),
        };
    }

    pub fn remove_buffer(&mut self, id: BufferId) {
        self.buffers.remove(id).unwrap();
    }

    pub fn describe_image(&mut self, desc: &ImageDesc) -> ImageId {
        self.images.insert(ImageResource::Described {
            desc: *desc,
            all_usage: ImageUsage::empty(),
        })
    }

    pub fn import_image(
        &mut self,
        desc: &ImageDesc,
        all_usage: ImageUsage,
        image: UniqueImage,
        current_usage: ImageUsage,
    ) -> ImageId {
        let image_view = self.resource_cache.get_image_view(desc, image);
        self.images.insert(ImageResource::Active {
            desc: *desc,
            _alloc: None,
            image,
            image_view,
            bindless_id: None,
            current_usage,
            all_usage_check: all_usage,
        })
    }

    pub fn create_image(&mut self, desc: &ImageDesc, all_usage: ImageUsage) -> ImageId {
        let all_usage_flags = all_usage.as_flags();
        let memory_property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let info = self.resource_cache.get_image_info(desc, all_usage_flags);
        let alloc = self.global_allocator.allocate(&info.mem_req, memory_property_flags);
        let image = self.resource_cache.get_image(desc, &info, &alloc, all_usage_flags);
        let image_view = self.resource_cache.get_image_view(desc, image);
        let bindless_id = self
            .bindless
            .as_mut()
            .and_then(|bindless| bindless.add_image(&desc, image_view.0, all_usage));
        self.images.insert(ImageResource::Active {
            desc: *desc,
            _alloc: Some(alloc),
            image,
            image_view,
            bindless_id,
            current_usage: ImageUsage::empty(),
            all_usage_check: all_usage,
        })
    }

    pub fn image_resource(&self, id: ImageId) -> &ImageResource {
        self.images.get(id).unwrap()
    }

    pub fn image_resource_mut(&mut self, id: ImageId) -> &mut ImageResource {
        self.images.get_mut(id).unwrap()
    }

    pub fn allocate_temporary_image(&mut self, id: ImageId, allocator: &mut Allocator) {
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
                    bindless_id: None,
                    image,
                    image_view,
                    current_usage: ImageUsage::empty(),
                    all_usage_check: *all_usage,
                }
            }
            _ => panic!("image is not temporary"),
        };
    }

    pub fn remove_image(&mut self, id: ImageId) {
        self.images.remove(id).unwrap();
    }

    pub fn transition_buffer_usage(
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

    pub fn transition_image_usage(
        &mut self,
        id: ImageId,
        new_usage: ImageUsage,
        context: &Context,
        cmd: vk::CommandBuffer,
    ) {
        self.images
            .get_mut(id)
            .unwrap()
            .transition_usage(new_usage, &context.device, cmd);
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        self.global_allocator.ui_stats_table_rows(ui, "global memory");

        ui.text("buffers");
        ui.next_column();
        ui.text(format!("{}", self.buffers.len()));
        ui.next_column();

        ui.text("images");
        ui.next_column();
        ui.text(format!("{}", self.images.len()));
        ui.next_column();

        self.resource_cache.ui_stats_table_rows(ui, "graph");

        if let Some(bindless) = self.bindless.as_ref() {
            bindless.ui_stats_table_rows(ui);
        }
    }
}
