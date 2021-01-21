use spark::{vk, Device};
use std::ops::{BitOr, BitOrAssign};
use std::slice;

struct SetBitIterator(u32);

impl Iterator for SetBitIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.0.trailing_zeros();
        if pos < 32 {
            let bit = 1 << pos;
            self.0 &= !bit;
            Some(bit)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u32);

impl BufferUsage {
    pub const TRANSFER_WRITE: BufferUsage = BufferUsage(0x1);
    pub const COMPUTE_STORAGE_READ: BufferUsage = BufferUsage(0x2);
    pub const VERTEX_BUFFER: BufferUsage = BufferUsage(0x4);
    pub const INDEX_BUFFER: BufferUsage = BufferUsage(0x8);
    pub const ACCELERATION_STRUCTURE_BUILD_INPUT: BufferUsage = BufferUsage(0x10);
    pub const ACCELERATION_STRUCTURE_BUILD_SCRATCH: BufferUsage = BufferUsage(0x20);
    pub const ACCELERATION_STRUCTURE_BUILD_WRITE: BufferUsage = BufferUsage(0x40);

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BufferUsage {
    pub fn as_flags(self) -> vk::BufferUsageFlags {
        SetBitIterator(self.0)
            .map(|bit| match Self(bit) {
                Self::TRANSFER_WRITE => vk::BufferUsageFlags::TRANSFER_DST,
                Self::COMPUTE_STORAGE_READ => vk::BufferUsageFlags::STORAGE_BUFFER,
                Self::VERTEX_BUFFER => vk::BufferUsageFlags::VERTEX_BUFFER,
                Self::INDEX_BUFFER => vk::BufferUsageFlags::INDEX_BUFFER,
                Self::ACCELERATION_STRUCTURE_BUILD_INPUT => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                }
                Self::ACCELERATION_STRUCTURE_BUILD_SCRATCH => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                }
                Self::ACCELERATION_STRUCTURE_BUILD_WRITE => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                }
                _ => unimplemented!(),
            })
            .fold(vk::BufferUsageFlags::empty(), |m, u| m | u)
    }

    pub fn as_stage_mask(self) -> vk::PipelineStageFlags {
        SetBitIterator(self.0)
            .map(|bit| match Self(bit) {
                Self::TRANSFER_WRITE => vk::PipelineStageFlags::TRANSFER,
                Self::COMPUTE_STORAGE_READ => vk::PipelineStageFlags::COMPUTE_SHADER,
                Self::VERTEX_BUFFER => vk::PipelineStageFlags::VERTEX_INPUT,
                Self::INDEX_BUFFER => vk::PipelineStageFlags::VERTEX_INPUT,
                Self::ACCELERATION_STRUCTURE_BUILD_INPUT => vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                Self::ACCELERATION_STRUCTURE_BUILD_SCRATCH => vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                Self::ACCELERATION_STRUCTURE_BUILD_WRITE => vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                _ => unimplemented!(),
            })
            .fold(vk::PipelineStageFlags::empty(), |m, u| m | u)
    }

    pub fn as_access_mask(self) -> vk::AccessFlags {
        SetBitIterator(self.0)
            .map(|bit| match Self(bit) {
                Self::TRANSFER_WRITE => vk::AccessFlags::TRANSFER_WRITE,
                Self::COMPUTE_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                Self::VERTEX_BUFFER => vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                Self::INDEX_BUFFER => vk::AccessFlags::INDEX_READ,
                Self::ACCELERATION_STRUCTURE_BUILD_INPUT => vk::AccessFlags::SHADER_READ,
                Self::ACCELERATION_STRUCTURE_BUILD_SCRATCH => {
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                }
                Self::ACCELERATION_STRUCTURE_BUILD_WRITE => vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                _ => unimplemented!(),
            })
            .fold(vk::AccessFlags::empty(), |m, u| m | u)
    }
}

pub fn emit_buffer_barrier(
    old_usage: BufferUsage,
    new_usage: BufferUsage,
    buffer: vk::Buffer,
    device: &Device,
    cmd: vk::CommandBuffer,
) {
    let buffer_memory_barrier = vk::BufferMemoryBarrier {
        src_access_mask: old_usage.as_access_mask(),
        dst_access_mask: new_usage.as_access_mask(),
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer: Some(buffer),
        offset: 0,
        size: vk::WHOLE_SIZE,
        ..Default::default()
    };
    let old_stage_mask = old_usage.as_stage_mask();
    let new_stage_mask = new_usage.as_stage_mask();
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            if old_stage_mask.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                old_stage_mask
            },
            if new_stage_mask.is_empty() {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                new_stage_mask
            },
            vk::DependencyFlags::empty(),
            &[],
            slice::from_ref(&buffer_memory_barrier),
            &[],
        )
    }
}

#[derive(Copy, Clone, Default)]
struct ImageUsageInfo {
    stage_mask: vk::PipelineStageFlags,
    access_mask: vk::AccessFlags,
    image_layout: vk::ImageLayout,
}

impl ImageUsageInfo {
    fn merge_with(&mut self, other: &Self) {
        self.stage_mask |= other.stage_mask;
        self.access_mask |= other.access_mask;
        if self.image_layout == vk::ImageLayout::UNDEFINED {
            self.image_layout = other.image_layout;
        } else if self.image_layout != other.image_layout {
            panic!(
                "cannot set image layouts {} and {} at the same time",
                self.image_layout, other.image_layout
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageUsage(u32);

impl ImageUsage {
    pub const TRANSFER_WRITE: ImageUsage = ImageUsage(0x1);
    pub const SWAPCHAIN: ImageUsage = ImageUsage(0x2);
    pub const COLOR_ATTACHMENT_WRITE: ImageUsage = ImageUsage(0x4);
    pub const FRAGMENT_SAMPLED: ImageUsage = ImageUsage(0x8);
    pub const COMPUTE_STORAGE_READ: ImageUsage = ImageUsage(0x10);
    pub const COMPUTE_STORAGE_WRITE: ImageUsage = ImageUsage(0x20);
    pub const TRANSIENT_COLOR_ATTACHMENT: ImageUsage = ImageUsage(0x40);
    pub const TRANSIENT_DEPTH_ATTACHMENT: ImageUsage = ImageUsage(0x80);

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for ImageUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ImageUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl ImageUsage {
    pub fn as_flags(self) -> vk::ImageUsageFlags {
        SetBitIterator(self.0)
            .map(|bit| match Self(bit) {
                Self::TRANSFER_WRITE => vk::ImageUsageFlags::TRANSFER_DST,
                Self::SWAPCHAIN => vk::ImageUsageFlags::empty(),
                Self::COLOR_ATTACHMENT_WRITE => vk::ImageUsageFlags::COLOR_ATTACHMENT,
                Self::FRAGMENT_SAMPLED => vk::ImageUsageFlags::SAMPLED,
                Self::COMPUTE_STORAGE_READ => vk::ImageUsageFlags::STORAGE,
                Self::COMPUTE_STORAGE_WRITE => vk::ImageUsageFlags::STORAGE,
                Self::TRANSIENT_COLOR_ATTACHMENT => {
                    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
                Self::TRANSIENT_DEPTH_ATTACHMENT => {
                    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                }
                _ => unimplemented!(),
            })
            .fold(vk::ImageUsageFlags::empty(), |m, u| m | u)
    }

    fn as_info(self) -> ImageUsageInfo {
        SetBitIterator(self.0)
            .map(|bit| match Self(bit) {
                Self::TRANSFER_WRITE => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::TRANSFER,
                    access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                },
                Self::SWAPCHAIN => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::empty(),
                    access_mask: vk::AccessFlags::empty(),
                    image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                },
                Self::COLOR_ATTACHMENT_WRITE => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                },
                Self::FRAGMENT_SAMPLED => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    access_mask: vk::AccessFlags::SHADER_READ,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                Self::COMPUTE_STORAGE_READ => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                    access_mask: vk::AccessFlags::SHADER_READ,
                    image_layout: vk::ImageLayout::GENERAL,
                },
                Self::COMPUTE_STORAGE_WRITE => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
                    access_mask: vk::AccessFlags::SHADER_WRITE,
                    image_layout: vk::ImageLayout::GENERAL,
                },
                Self::TRANSIENT_COLOR_ATTACHMENT | Self::TRANSIENT_DEPTH_ATTACHMENT => ImageUsageInfo {
                    stage_mask: vk::PipelineStageFlags::empty(),
                    access_mask: vk::AccessFlags::empty(),
                    image_layout: vk::ImageLayout::UNDEFINED,
                },
                _ => unimplemented!(),
            })
            .fold(ImageUsageInfo::default(), |mut m, u| {
                m.merge_with(&u);
                m
            })
    }
}

pub fn emit_image_barrier(
    old_usage: ImageUsage,
    new_usage: ImageUsage,
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    device: &Device,
    cmd: vk::CommandBuffer,
) {
    let old_info = old_usage.as_info();
    let new_info = new_usage.as_info();

    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: old_info.access_mask,
        dst_access_mask: new_info.access_mask,
        old_layout: old_info.image_layout,
        new_layout: new_info.image_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: Some(image),
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        },
        ..Default::default()
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            if old_info.stage_mask.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                old_info.stage_mask
            },
            if new_info.stage_mask.is_empty() {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                new_info.stage_mask
            },
            vk::DependencyFlags::empty(),
            &[],
            &[],
            slice::from_ref(&image_memory_barrier),
        )
    }
}
