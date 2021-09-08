use crate::context::*;
use spark::{vk, Builder};
use std::slice;

struct CommandBufferSemaphores {
    image_available: vk::Semaphore,
    rendering_finished: vk::Semaphore,
}

struct CommandBufferSet {
    pool: vk::CommandPool,
    pre_swapchain_cmd: vk::CommandBuffer,
    post_swapchain_cmd: vk::CommandBuffer,
    fence: vk::Fence,
    semaphores: Option<CommandBufferSemaphores>,
}

pub struct CommandBufferAcquireResult {
    pub pre_swapchain_cmd: vk::CommandBuffer,
    pub post_swapchain_cmd: vk::CommandBuffer,
    pub image_available_semaphore: Option<vk::Semaphore>,
}

impl CommandBufferSet {
    fn new(context: &Context) -> Self {
        let device = &context.device;

        let pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index: context.queue_family_index,
                ..Default::default()
            };
            unsafe { device.create_command_pool(&command_pool_create_info, None) }.unwrap()
        };

        let (pre_swapchain_cmd, post_swapchain_cmd) = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: Some(pool),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 2,
                ..Default::default()
            };

            let command_buffers: [vk::CommandBuffer; 2] =
                unsafe { device.allocate_command_buffers_array(&command_buffer_allocate_info) }.unwrap();
            (command_buffers[0], command_buffers[1])
        };

        let fence = {
            let fence_create_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };
            unsafe { device.create_fence(&fence_create_info, None) }.unwrap()
        };

        let semaphores = context.surface.map(|_| CommandBufferSemaphores {
            image_available: unsafe { device.create_semaphore(&Default::default(), None) }.unwrap(),
            rendering_finished: unsafe { device.create_semaphore(&Default::default(), None) }.unwrap(),
        });

        Self {
            pool,
            pre_swapchain_cmd,
            post_swapchain_cmd,
            fence,
            semaphores,
        }
    }

    fn wait_for_fence(&self, context: &Context) {
        let timeout_ns = 1000 * 1000 * 1000;
        loop {
            let res = unsafe {
                context
                    .device
                    .wait_for_fences(slice::from_ref(&self.fence), true, timeout_ns)
            };
            match res {
                Ok(_) => break,
                Err(vk::Result::TIMEOUT) => {}
                Err(err_code) => panic!("failed to wait for fence {}", err_code),
            }
        }
    }
}

pub struct CommandBufferPool {
    context: SharedContext,
    sets: [CommandBufferSet; Self::COUNT],
    index: usize,
}

impl CommandBufferPool {
    pub const COUNT: usize = 2;

    pub fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            sets: [CommandBufferSet::new(context), CommandBufferSet::new(context)],
            index: 0,
        }
    }

    pub fn acquire(&mut self) -> CommandBufferAcquireResult {
        self.index = (self.index + 1) % Self::COUNT;

        let set = &self.sets[self.index];

        set.wait_for_fence(&self.context);

        unsafe { self.context.device.reset_fences(slice::from_ref(&set.fence)) }.unwrap();

        unsafe {
            self.context
                .device
                .reset_command_pool(set.pool, vk::CommandPoolResetFlags::empty())
        }
        .unwrap();

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            self.context
                .device
                .begin_command_buffer(set.pre_swapchain_cmd, &command_buffer_begin_info)
        }
        .unwrap();
        unsafe {
            self.context
                .device
                .begin_command_buffer(set.post_swapchain_cmd, &command_buffer_begin_info)
        }
        .unwrap();

        CommandBufferAcquireResult {
            pre_swapchain_cmd: set.pre_swapchain_cmd,
            image_available_semaphore: set.semaphores.as_ref().map(|s| s.image_available),
            post_swapchain_cmd: set.post_swapchain_cmd,
        }
    }

    pub fn submit(&self) -> Option<vk::Semaphore> {
        let set = &self.sets[self.index];

        unsafe { self.context.device.end_command_buffer(set.pre_swapchain_cmd) }.unwrap();
        unsafe { self.context.device.end_command_buffer(set.post_swapchain_cmd) }.unwrap();

        let submit_info = [
            *vk::SubmitInfo::builder().p_command_buffers(slice::from_ref(&set.pre_swapchain_cmd)),
            if let Some(semaphores) = set.semaphores.as_ref() {
                *vk::SubmitInfo::builder()
                    .p_wait_semaphores(
                        slice::from_ref(&semaphores.image_available),
                        slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    )
                    .p_command_buffers(slice::from_ref(&set.post_swapchain_cmd))
                    .p_signal_semaphores(slice::from_ref(&semaphores.rendering_finished))
            } else {
                *vk::SubmitInfo::builder().p_command_buffers(slice::from_ref(&set.post_swapchain_cmd))
            },
        ];

        unsafe {
            self.context
                .device
                .queue_submit(self.context.queue, &submit_info, Some(set.fence))
        }
        .unwrap();

        set.semaphores.as_ref().map(|s| s.rendering_finished)
    }

    pub fn wait_after_submit(&self) {
        let set = &self.sets[self.index];
        set.wait_for_fence(&self.context);
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        let device = &self.context.device;
        for set in self.sets.iter() {
            unsafe {
                if let Some(semaphores) = set.semaphores.as_ref() {
                    device.destroy_semaphore(Some(semaphores.rendering_finished), None);
                    device.destroy_semaphore(Some(semaphores.image_available), None);
                }
                device.destroy_fence(Some(set.fence), None);
                device.free_command_buffers(set.pool, &[set.pre_swapchain_cmd, set.post_swapchain_cmd]);
                device.destroy_command_pool(Some(set.pool), None);
            }
        }
    }
}
