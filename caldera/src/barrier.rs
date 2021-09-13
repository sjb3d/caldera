use bytemuck::Contiguous;
use spark::{vk, Device};
use std::ops::{BitOr, BitOrAssign};
use std::slice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessCategory(u32);

impl AccessCategory {
    pub const READ: AccessCategory = AccessCategory(0x1);
    pub const WRITE: AccessCategory = AccessCategory(0x2);
    // TODO: atomic

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn supports_overlap(&self) -> bool {
        self.0.count_ones() < 2
    }
}

impl BitOr for AccessCategory {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for AccessCategory {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

macro_rules! usage_impl {
    ($usage:ident, $usage_bit:ident, $usage_bit_iter:ident; $($i:ident),+ $(,)*) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Contiguous)]
        #[allow(non_camel_case_types)]
        enum $usage_bit {
            $($i),+
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $usage(u32);

        impl $usage {
            $(pub const $i: $usage = $usage(1 << ($usage_bit::$i as u32));)+

            pub fn empty() -> Self {
                Self(0)
            }

            pub fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            fn iter_set_bits(self) -> $usage_bit_iter {
                $usage_bit_iter(self)
            }
        }

        impl BitOr for $usage {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl BitOrAssign for $usage {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        struct $usage_bit_iter($usage);

        impl Iterator for $usage_bit_iter {
            type Item = $usage_bit;

            fn next(&mut self) -> Option<Self::Item> {
                let pos = self.0.0.trailing_zeros();
                if pos < 32 {
                    let bit = 1 << pos;
                    self.0.0 &= !bit;
                    Some($usage_bit::from_integer(pos).unwrap())
                } else {
                    None
                }
            }
        }
    };
}

usage_impl! {
    BufferUsage, BufferUsageBit, BufferUsageBitIterator;
    TRANSFER_WRITE,
    COMPUTE_STORAGE_READ,
    COMPUTE_STORAGE_WRITE,
    VERTEX_BUFFER,
    VERTEX_STORAGE_READ,
    INDEX_BUFFER,
    ACCELERATION_STRUCTURE_BUILD_INPUT,
    ACCELERATION_STRUCTURE_BUILD_SCRATCH,
    ACCELERATION_STRUCTURE_READ,
    ACCELERATION_STRUCTURE_WRITE,
    RAY_TRACING_ACCELERATION_STRUCTURE,
    RAY_TRACING_SHADER_BINDING_TABLE,
    RAY_TRACING_STORAGE_READ,
    TASK_STORAGE_READ,
    MESH_STORAGE_READ,
    HOST_READ,
}

impl BufferUsage {
    pub fn as_flags(self) -> vk::BufferUsageFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                BufferUsageBit::TRANSFER_WRITE => vk::BufferUsageFlags::TRANSFER_DST,
                BufferUsageBit::COMPUTE_STORAGE_READ => vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsageBit::COMPUTE_STORAGE_WRITE => vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsageBit::VERTEX_BUFFER => vk::BufferUsageFlags::VERTEX_BUFFER,
                BufferUsageBit::VERTEX_STORAGE_READ => vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsageBit::INDEX_BUFFER => vk::BufferUsageFlags::INDEX_BUFFER,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                }
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_SCRATCH => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::STORAGE_BUFFER
                }
                BufferUsageBit::ACCELERATION_STRUCTURE_READ
                | BufferUsageBit::ACCELERATION_STRUCTURE_WRITE
                | BufferUsageBit::RAY_TRACING_ACCELERATION_STRUCTURE => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                }
                BufferUsageBit::RAY_TRACING_SHADER_BINDING_TABLE => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                }
                BufferUsageBit::RAY_TRACING_STORAGE_READ => {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::STORAGE_BUFFER
                }
                BufferUsageBit::TASK_STORAGE_READ => vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsageBit::MESH_STORAGE_READ => vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferUsageBit::HOST_READ => vk::BufferUsageFlags::empty(),
            })
            .fold(vk::BufferUsageFlags::empty(), |m, u| m | u)
    }

    pub fn as_stage_mask(self) -> vk::PipelineStageFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                BufferUsageBit::TRANSFER_WRITE => vk::PipelineStageFlags::TRANSFER,
                BufferUsageBit::COMPUTE_STORAGE_READ => vk::PipelineStageFlags::COMPUTE_SHADER,
                BufferUsageBit::COMPUTE_STORAGE_WRITE => vk::PipelineStageFlags::COMPUTE_SHADER,
                BufferUsageBit::VERTEX_BUFFER => vk::PipelineStageFlags::VERTEX_INPUT,
                BufferUsageBit::VERTEX_STORAGE_READ => vk::PipelineStageFlags::VERTEX_SHADER,
                BufferUsageBit::INDEX_BUFFER => vk::PipelineStageFlags::VERTEX_INPUT,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT => {
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                }
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_SCRATCH => {
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                }
                BufferUsageBit::ACCELERATION_STRUCTURE_READ => vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                BufferUsageBit::ACCELERATION_STRUCTURE_WRITE => {
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                }
                BufferUsageBit::RAY_TRACING_ACCELERATION_STRUCTURE => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                BufferUsageBit::RAY_TRACING_SHADER_BINDING_TABLE => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                BufferUsageBit::RAY_TRACING_STORAGE_READ => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                BufferUsageBit::TASK_STORAGE_READ => vk::PipelineStageFlags::TASK_SHADER_NV,
                BufferUsageBit::MESH_STORAGE_READ => vk::PipelineStageFlags::MESH_SHADER_NV,
                BufferUsageBit::HOST_READ => vk::PipelineStageFlags::HOST,
            })
            .fold(vk::PipelineStageFlags::empty(), |m, u| m | u)
    }

    pub fn as_access_mask(self) -> vk::AccessFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                BufferUsageBit::TRANSFER_WRITE => vk::AccessFlags::TRANSFER_WRITE,
                BufferUsageBit::COMPUTE_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::COMPUTE_STORAGE_WRITE => vk::AccessFlags::SHADER_WRITE,
                BufferUsageBit::VERTEX_BUFFER => vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                BufferUsageBit::VERTEX_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::INDEX_BUFFER => vk::AccessFlags::INDEX_READ,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_SCRATCH => {
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                }
                BufferUsageBit::ACCELERATION_STRUCTURE_READ => vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                BufferUsageBit::ACCELERATION_STRUCTURE_WRITE => vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                BufferUsageBit::RAY_TRACING_ACCELERATION_STRUCTURE => vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                BufferUsageBit::RAY_TRACING_SHADER_BINDING_TABLE => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::RAY_TRACING_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::TASK_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::MESH_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                BufferUsageBit::HOST_READ => vk::AccessFlags::HOST_READ,
            })
            .fold(vk::AccessFlags::empty(), |m, u| m | u)
    }

    pub fn as_access_category(self) -> AccessCategory {
        self.iter_set_bits()
            .map(|bit| match bit {
                BufferUsageBit::TRANSFER_WRITE => AccessCategory::WRITE,
                BufferUsageBit::COMPUTE_STORAGE_READ => AccessCategory::READ,
                BufferUsageBit::COMPUTE_STORAGE_WRITE => AccessCategory::WRITE,
                BufferUsageBit::VERTEX_BUFFER => AccessCategory::READ,
                BufferUsageBit::VERTEX_STORAGE_READ => AccessCategory::READ,
                BufferUsageBit::INDEX_BUFFER => AccessCategory::READ,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_INPUT => AccessCategory::READ,
                BufferUsageBit::ACCELERATION_STRUCTURE_BUILD_SCRATCH => AccessCategory::READ | AccessCategory::WRITE,
                BufferUsageBit::ACCELERATION_STRUCTURE_READ => AccessCategory::READ,
                BufferUsageBit::ACCELERATION_STRUCTURE_WRITE => AccessCategory::WRITE,
                BufferUsageBit::RAY_TRACING_ACCELERATION_STRUCTURE => AccessCategory::READ,
                BufferUsageBit::RAY_TRACING_SHADER_BINDING_TABLE => AccessCategory::READ,
                BufferUsageBit::RAY_TRACING_STORAGE_READ => AccessCategory::READ,
                BufferUsageBit::TASK_STORAGE_READ => AccessCategory::READ,
                BufferUsageBit::MESH_STORAGE_READ => AccessCategory::READ,
                BufferUsageBit::HOST_READ => AccessCategory::READ,
            })
            .fold(AccessCategory::empty(), |m, u| m | u)
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
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                old_stage_mask
            },
            if new_stage_mask.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
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

usage_impl! {
    ImageUsage, ImageUsageBit, ImageUsageBitIterator;
    TRANSFER_WRITE,
    SWAPCHAIN,
    COLOR_ATTACHMENT_WRITE,
    FRAGMENT_STORAGE_READ,
    FRAGMENT_SAMPLED,
    COMPUTE_STORAGE_READ,
    COMPUTE_STORAGE_WRITE,
    COMPUTE_SAMPLED,
    TRANSIENT_COLOR_ATTACHMENT,
    TRANSIENT_DEPTH_ATTACHMENT,
    RAY_TRACING_STORAGE_READ,
    RAY_TRACING_STORAGE_WRITE,
    RAY_TRACING_SAMPLED,
}

impl ImageUsage {
    pub fn as_flags(self) -> vk::ImageUsageFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                ImageUsageBit::TRANSFER_WRITE => vk::ImageUsageFlags::TRANSFER_DST,
                ImageUsageBit::SWAPCHAIN => vk::ImageUsageFlags::empty(),
                ImageUsageBit::COLOR_ATTACHMENT_WRITE => vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ImageUsageBit::FRAGMENT_STORAGE_READ => vk::ImageUsageFlags::STORAGE,
                ImageUsageBit::FRAGMENT_SAMPLED => vk::ImageUsageFlags::SAMPLED,
                ImageUsageBit::COMPUTE_STORAGE_READ => vk::ImageUsageFlags::STORAGE,
                ImageUsageBit::COMPUTE_STORAGE_WRITE => vk::ImageUsageFlags::STORAGE,
                ImageUsageBit::COMPUTE_SAMPLED => vk::ImageUsageFlags::SAMPLED,
                ImageUsageBit::TRANSIENT_COLOR_ATTACHMENT => {
                    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT
                }
                ImageUsageBit::TRANSIENT_DEPTH_ATTACHMENT => {
                    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                }
                ImageUsageBit::RAY_TRACING_STORAGE_READ => vk::ImageUsageFlags::STORAGE,
                ImageUsageBit::RAY_TRACING_STORAGE_WRITE => vk::ImageUsageFlags::STORAGE,
                ImageUsageBit::RAY_TRACING_SAMPLED => vk::ImageUsageFlags::SAMPLED,
            })
            .fold(vk::ImageUsageFlags::empty(), |m, u| m | u)
    }

    pub fn as_stage_mask(self) -> vk::PipelineStageFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                ImageUsageBit::TRANSFER_WRITE => vk::PipelineStageFlags::TRANSFER,
                ImageUsageBit::SWAPCHAIN => vk::PipelineStageFlags::empty(),
                ImageUsageBit::COLOR_ATTACHMENT_WRITE => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ImageUsageBit::FRAGMENT_STORAGE_READ => vk::PipelineStageFlags::FRAGMENT_SHADER,
                ImageUsageBit::FRAGMENT_SAMPLED => vk::PipelineStageFlags::FRAGMENT_SHADER,
                ImageUsageBit::COMPUTE_STORAGE_READ => vk::PipelineStageFlags::COMPUTE_SHADER,
                ImageUsageBit::COMPUTE_STORAGE_WRITE => vk::PipelineStageFlags::COMPUTE_SHADER,
                ImageUsageBit::COMPUTE_SAMPLED => vk::PipelineStageFlags::COMPUTE_SHADER,
                ImageUsageBit::TRANSIENT_COLOR_ATTACHMENT | ImageUsageBit::TRANSIENT_DEPTH_ATTACHMENT => {
                    vk::PipelineStageFlags::empty()
                }
                ImageUsageBit::RAY_TRACING_STORAGE_READ => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                ImageUsageBit::RAY_TRACING_STORAGE_WRITE => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                ImageUsageBit::RAY_TRACING_SAMPLED => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            })
            .fold(vk::PipelineStageFlags::empty(), |m, u| m | u)
    }

    pub fn as_access_mask(self) -> vk::AccessFlags {
        self.iter_set_bits()
            .map(|bit| match bit {
                ImageUsageBit::TRANSFER_WRITE => vk::AccessFlags::TRANSFER_WRITE,
                ImageUsageBit::SWAPCHAIN => vk::AccessFlags::empty(),
                ImageUsageBit::COLOR_ATTACHMENT_WRITE => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ImageUsageBit::FRAGMENT_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                ImageUsageBit::FRAGMENT_SAMPLED => vk::AccessFlags::SHADER_READ,
                ImageUsageBit::COMPUTE_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                ImageUsageBit::COMPUTE_STORAGE_WRITE => vk::AccessFlags::SHADER_WRITE,
                ImageUsageBit::COMPUTE_SAMPLED => vk::AccessFlags::SHADER_READ,
                ImageUsageBit::TRANSIENT_COLOR_ATTACHMENT | ImageUsageBit::TRANSIENT_DEPTH_ATTACHMENT => {
                    vk::AccessFlags::empty()
                }
                ImageUsageBit::RAY_TRACING_STORAGE_READ => vk::AccessFlags::SHADER_READ,
                ImageUsageBit::RAY_TRACING_STORAGE_WRITE => vk::AccessFlags::SHADER_WRITE,
                ImageUsageBit::RAY_TRACING_SAMPLED => vk::AccessFlags::SHADER_READ,
            })
            .fold(vk::AccessFlags::empty(), |m, u| m | u)
    }

    pub fn as_image_layout(self) -> vk::ImageLayout {
        self.iter_set_bits()
            .map(|bit| match bit {
                ImageUsageBit::TRANSFER_WRITE => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageUsageBit::SWAPCHAIN => vk::ImageLayout::PRESENT_SRC_KHR,
                ImageUsageBit::COLOR_ATTACHMENT_WRITE => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ImageUsageBit::FRAGMENT_STORAGE_READ => vk::ImageLayout::GENERAL,
                ImageUsageBit::FRAGMENT_SAMPLED => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ImageUsageBit::COMPUTE_STORAGE_READ => vk::ImageLayout::GENERAL,
                ImageUsageBit::COMPUTE_STORAGE_WRITE => vk::ImageLayout::GENERAL,
                ImageUsageBit::COMPUTE_SAMPLED => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ImageUsageBit::TRANSIENT_COLOR_ATTACHMENT | ImageUsageBit::TRANSIENT_DEPTH_ATTACHMENT => {
                    vk::ImageLayout::UNDEFINED
                }
                ImageUsageBit::RAY_TRACING_STORAGE_READ => vk::ImageLayout::GENERAL,
                ImageUsageBit::RAY_TRACING_STORAGE_WRITE => vk::ImageLayout::GENERAL,
                ImageUsageBit::RAY_TRACING_SAMPLED => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .fold(vk::ImageLayout::UNDEFINED, |m, u| {
                if m == u {
                    m
                } else if m == vk::ImageLayout::UNDEFINED {
                    u
                } else {
                    panic!("cannot set image layouts {} and {} at the same time", m, u)
                }
            })
    }

    pub fn as_access_category(self) -> AccessCategory {
        self.iter_set_bits()
            .map(|bit| match bit {
                ImageUsageBit::TRANSFER_WRITE => AccessCategory::WRITE,
                ImageUsageBit::SWAPCHAIN => AccessCategory::READ,
                ImageUsageBit::COLOR_ATTACHMENT_WRITE => AccessCategory::WRITE,
                ImageUsageBit::FRAGMENT_STORAGE_READ => AccessCategory::READ,
                ImageUsageBit::FRAGMENT_SAMPLED => AccessCategory::READ,
                ImageUsageBit::COMPUTE_STORAGE_READ => AccessCategory::READ,
                ImageUsageBit::COMPUTE_STORAGE_WRITE => AccessCategory::WRITE,
                ImageUsageBit::COMPUTE_SAMPLED => AccessCategory::READ,
                ImageUsageBit::TRANSIENT_COLOR_ATTACHMENT | ImageUsageBit::TRANSIENT_DEPTH_ATTACHMENT => {
                    AccessCategory::READ | AccessCategory::WRITE
                }
                ImageUsageBit::RAY_TRACING_STORAGE_READ => AccessCategory::READ,
                ImageUsageBit::RAY_TRACING_STORAGE_WRITE => AccessCategory::WRITE,
                ImageUsageBit::RAY_TRACING_SAMPLED => AccessCategory::READ,
            })
            .fold(AccessCategory::empty(), |m, u| m | u)
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
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: old_usage.as_access_mask(),
        dst_access_mask: new_usage.as_access_mask(),
        old_layout: old_usage.as_image_layout(),
        new_layout: new_usage.as_image_layout(),
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
    let old_stage_mask = old_usage.as_stage_mask();
    let new_stage_mask = new_usage.as_stage_mask();
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            if old_stage_mask.is_empty() {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                old_stage_mask
            },
            if new_stage_mask.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                new_stage_mask
            },
            vk::DependencyFlags::empty(),
            &[],
            &[],
            slice::from_ref(&image_memory_barrier),
        )
    }
}
