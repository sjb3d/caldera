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

macro_rules! base_usage_impl {
    ($usage:ident, $usage_bit:ident, $usage_bit_iter:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $usage(u32);

        impl $usage {
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
                let pos = self.0 .0.trailing_zeros();
                if pos < 32 {
                    let bit = 1 << pos;
                    self.0 .0 &= !bit;
                    Some($usage_bit::from_integer(pos).unwrap())
                } else {
                    None
                }
            }
        }
    };
}

macro_rules! buffer_usage_impl {
    ($(($name:ident, $usage_mask:expr, $stage_mask:expr, $access_mask:expr, $access_category:expr)),+ $(,)?) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Contiguous)]
        #[allow(non_camel_case_types)]
        enum BufferUsageBit {
            $($name),+
        }

        base_usage_impl!(BufferUsage, BufferUsageBit, BufferUsageBitIterator);

        impl BufferUsage {
            $(pub const $name: BufferUsage = BufferUsage(1 << (BufferUsageBit::$name as u32));)+

            pub fn as_flags(self) -> vk::BufferUsageFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(BufferUsageBit::$name => $usage_mask),+
                    })
                    .fold(vk::BufferUsageFlags::empty(), |m, u| m | u)
            }

            pub fn as_stage_mask(self) -> vk::PipelineStageFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(BufferUsageBit::$name => $stage_mask),+
                    })
                    .fold(vk::PipelineStageFlags::empty(), |m, u| m | u)
            }

            pub fn as_access_mask(self) -> vk::AccessFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(BufferUsageBit::$name => $access_mask),+
                    })
                    .fold(vk::AccessFlags::empty(), |m, u| m | u)
            }

            pub fn as_access_category(self) -> AccessCategory {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(BufferUsageBit::$name => $access_category),+
                    })
                    .fold(AccessCategory::empty(), |m, u| m | u)
            }
        }
    }
}

buffer_usage_impl! {
    (
        TRANSFER_WRITE,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_WRITE,
        AccessCategory::WRITE
    ),
    (
        COMPUTE_STORAGE_READ,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        COMPUTE_STORAGE_WRITE,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        AccessCategory::WRITE
    ),
    (
        VERTEX_BUFFER,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::PipelineStageFlags::VERTEX_INPUT,
        vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        AccessCategory::READ
    ),
    (
        VERTEX_STORAGE_READ,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::VERTEX_SHADER,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        INDEX_BUFFER,
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::PipelineStageFlags::VERTEX_INPUT,
        vk::AccessFlags::INDEX_READ,
        AccessCategory::READ
    ),
    (
        ACCELERATION_STRUCTURE_BUILD_INPUT,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        ACCELERATION_STRUCTURE_BUILD_SCRATCH,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        AccessCategory::READ | AccessCategory::WRITE
    ),
    (
        ACCELERATION_STRUCTURE_READ,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        AccessCategory::READ
    ),
    (
        ACCELERATION_STRUCTURE_WRITE,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        AccessCategory::WRITE
    ),
    (
        RAY_TRACING_ACCELERATION_STRUCTURE,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        AccessCategory::READ
    ),
    (
        RAY_TRACING_SHADER_BINDING_TABLE,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        RAY_TRACING_STORAGE_READ,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        TASK_STORAGE_READ,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::TASK_SHADER_NV,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        MESH_STORAGE_READ,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::PipelineStageFlags::MESH_SHADER_NV,
        vk::AccessFlags::SHADER_READ,
        AccessCategory::READ
    ),
    (
        HOST_READ,
        vk::BufferUsageFlags::empty(),
        vk::PipelineStageFlags::HOST,
        vk::AccessFlags::HOST_READ,
        AccessCategory::READ
    ),
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

macro_rules! image_usage_impl {
    ($(($name:ident, $usage_mask:expr, $stage_mask:expr, $access_mask:expr, $image_layout:expr, $access_category:expr)),+ $(,)?) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Contiguous)]
        #[allow(non_camel_case_types)]
        enum ImageUsageBit {
            $($name),+
        }

        base_usage_impl!(ImageUsage, ImageUsageBit, ImageUsageBitIterator);

        impl ImageUsage {
            $(pub const $name: ImageUsage = ImageUsage(1 << (ImageUsageBit::$name as u32));)+

            pub fn as_flags(self) -> vk::ImageUsageFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(ImageUsageBit::$name => $usage_mask),+
                    })
                    .fold(vk::ImageUsageFlags::empty(), |m, u| m | u)
            }

            pub fn as_stage_mask(self) -> vk::PipelineStageFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(ImageUsageBit::$name => $stage_mask),+
                    })
                    .fold(vk::PipelineStageFlags::empty(), |m, u| m | u)
            }

            pub fn as_access_mask(self) -> vk::AccessFlags {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(ImageUsageBit::$name => $access_mask),+
                    })
                    .fold(vk::AccessFlags::empty(), |m, u| m | u)
            }

            pub fn as_image_layout(self) -> vk::ImageLayout {
                self.iter_set_bits()
                    .map(|bit| match bit {
                        $(ImageUsageBit::$name => $image_layout),+
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
                        $(ImageUsageBit::$name => $access_category),+
                    })
                    .fold(AccessCategory::empty(), |m, u| m | u)
            }
        }
    }
}

image_usage_impl! {
    (
        TRANSFER_WRITE,
        vk::ImageUsageFlags::TRANSFER_DST,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        AccessCategory::WRITE
    ),
    (
        SWAPCHAIN,
        vk::ImageUsageFlags::empty(),
        vk::PipelineStageFlags::empty(),
        vk::AccessFlags::empty(),
        vk::ImageLayout::PRESENT_SRC_KHR,
        AccessCategory::READ
    ),
    (
        COLOR_ATTACHMENT_WRITE,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        AccessCategory::WRITE
    ),
    (
        FRAGMENT_STORAGE_READ,
        vk::ImageUsageFlags::STORAGE,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::GENERAL,
        AccessCategory::READ
    ),
    (
        FRAGMENT_SAMPLED,
        vk::ImageUsageFlags::SAMPLED,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AccessCategory::READ
    ),
    (
        COMPUTE_STORAGE_READ,
        vk::ImageUsageFlags::STORAGE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::GENERAL,
        AccessCategory::READ
    ),
    (
        COMPUTE_STORAGE_WRITE,
        vk::ImageUsageFlags::STORAGE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        vk::ImageLayout::GENERAL,
        AccessCategory::WRITE
    ),
    (
        COMPUTE_SAMPLED,
        vk::ImageUsageFlags::SAMPLED,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AccessCategory::READ
    ),
    (
        TRANSIENT_COLOR_ATTACHMENT,
        vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::PipelineStageFlags::empty(),
        vk::AccessFlags::empty(),
        vk::ImageLayout::UNDEFINED,
        AccessCategory::READ | AccessCategory::WRITE
    ),
    (
        TRANSIENT_DEPTH_ATTACHMENT,
        vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::PipelineStageFlags::empty(),
        vk::AccessFlags::empty(),
        vk::ImageLayout::UNDEFINED,
        AccessCategory::READ | AccessCategory::WRITE
    ),
    (
        RAY_TRACING_STORAGE_READ,
        vk::ImageUsageFlags::STORAGE,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::GENERAL,
        AccessCategory::READ
    ),
    (
        RAY_TRACING_STORAGE_WRITE,
        vk::ImageUsageFlags::STORAGE,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::SHADER_WRITE,
        vk::ImageLayout::GENERAL,
        AccessCategory::WRITE
    ),
    (
        RAY_TRACING_SAMPLED,
        vk::ImageUsageFlags::SAMPLED,
        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AccessCategory::READ
    ),
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
