use crate::allocator::*;
use crate::context::*;
use crate::maths::*;
use arrayvec::ArrayVec;
use imgui::Ui;
use spark::{vk, Builder, Device};
use std::collections::HashMap;
use std::slice;
use std::sync::Arc;

pub trait FormatExt {
    fn bits_per_element(&self) -> usize;
}

impl FormatExt for vk::Format {
    fn bits_per_element(&self) -> usize {
        match *self {
            vk::Format::BC1_RGB_SRGB_BLOCK => 4,
            vk::Format::R8G8B8A8_SRGB | vk::Format::R16G16_UNORM | vk::Format::R16G16_UINT | vk::Format::R32_SFLOAT => {
                32
            }
            vk::Format::R16G16B16A16_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32G32_UINT => 64,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: usize,
}

impl BufferDesc {
    pub fn new(size: usize) -> Self {
        BufferDesc { size }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferInfo {
    pub mem_req: vk::MemoryRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    pub size: UVec2,
    pub layer_count: u32,
    pub format: vk::Format,
    pub aspect_mask: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,
}

impl ImageDesc {
    pub fn new_2d(size: UVec2, format: vk::Format, aspect_mask: vk::ImageAspectFlags) -> Self {
        ImageDesc {
            size,
            layer_count: 1,
            format,
            aspect_mask,
            samples: vk::SampleCountFlags::N1,
        }
    }

    pub fn with_layer_count(mut self, layer_count: u32) -> Self {
        self.layer_count = layer_count;
        self
    }

    pub fn with_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    pub fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.size.x,
            height: self.size.y,
        }
    }

    pub fn is_array(&self) -> bool {
        self.layer_count > 1
    }

    pub(crate) fn view_type(&self) -> vk::ImageViewType {
        if self.layer_count > 1 {
            vk::ImageViewType::N2D_ARRAY
        } else {
            vk::ImageViewType::N2D
        }
    }

    pub(crate) fn staging_size(&self) -> usize {
        assert_eq!(self.samples, vk::SampleCountFlags::N1);
        (self.size.x as usize) * (self.size.y as usize) * (self.layer_count as usize) * self.format.bits_per_element()
            / 8
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    pub mem_req: vk::MemoryRequirements,
}

trait DeviceGraphExt {
    fn create_buffer_from_desc(
        &self,
        desc: &BufferDesc,
        usage_flags: vk::BufferUsageFlags,
    ) -> spark::Result<vk::Buffer>;

    fn create_image_from_desc(&self, desc: &ImageDesc, usage_flags: vk::ImageUsageFlags) -> spark::Result<vk::Image>;
}

impl DeviceGraphExt for Device {
    fn create_buffer_from_desc(
        &self,
        desc: &BufferDesc,
        usage_flags: vk::BufferUsageFlags,
    ) -> spark::Result<vk::Buffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size: desc.size as vk::DeviceSize,
            usage: usage_flags,
            ..Default::default()
        };
        unsafe { self.create_buffer(&buffer_create_info, None) }
    }

    fn create_image_from_desc(&self, desc: &ImageDesc, usage_flags: vk::ImageUsageFlags) -> spark::Result<vk::Image> {
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::N2D,
            format: desc.format,
            extent: vk::Extent3D {
                width: desc.size.x,
                height: desc.size.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: desc.layer_count,
            samples: desc.samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: usage_flags,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        unsafe { self.create_image(&image_create_info, None) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RenderPassKey {
    color_format: vk::Format,
    depth_format: Option<vk::Format>,
    samples: vk::SampleCountFlags,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BufferInfoKey {
    desc: BufferDesc,
    all_usage_flags: vk::BufferUsageFlags,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BufferKey {
    desc: BufferDesc,
    all_usage_flags: vk::BufferUsageFlags,
    alloc: Alloc,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ImageInfoKey {
    desc: ImageDesc,
    all_usage_flags: vk::ImageUsageFlags,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ImageKey {
    desc: ImageDesc,
    all_usage_flags: vk::ImageUsageFlags,
    alloc: Alloc,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ImageViewKey {
    image: UniqueImage,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FramebufferKey {
    render_pass: UniqueRenderPass,
    size: UVec2,
    color_output_image_view: UniqueImageView,
    color_temp_image_view: Option<UniqueImageView>,
    depth_temp_image_view: Option<UniqueImageView>,
}

pub(crate) struct ResourceCache {
    context: Arc<Context>,
    buffer_info: HashMap<BufferInfoKey, BufferInfo>,
    buffer: HashMap<BufferKey, UniqueBuffer>,
    image_info: HashMap<ImageInfoKey, ImageInfo>,
    image: HashMap<ImageKey, UniqueImage>,
    image_view: HashMap<ImageViewKey, UniqueImageView>,
}

pub(crate) struct RenderCache {
    context: Arc<Context>,
    render_pass: HashMap<RenderPassKey, UniqueRenderPass>,
    framebuffer: HashMap<FramebufferKey, UniqueFramebuffer>,
}

impl ResourceCache {
    pub fn new(context: &Arc<Context>) -> Self {
        Self {
            context: Arc::clone(&context),
            buffer_info: HashMap::new(),
            buffer: HashMap::new(),
            image_info: HashMap::new(),
            image: HashMap::new(),
            image_view: HashMap::new(),
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui, name: &str) {
        ui.text(format!("{} buffer cache", name));
        ui.next_column();
        ui.text(format!("{},{}", self.buffer_info.len(), self.buffer.len()));
        ui.next_column();

        ui.text(format!("{} image cache", name));
        ui.next_column();
        ui.text(format!(
            "{},{},{}",
            self.image_info.len(),
            self.image.len(),
            self.image_view.len()
        ));
        ui.next_column();
    }

    pub fn get_buffer_info(&mut self, desc: &BufferDesc, all_usage_flags: vk::BufferUsageFlags) -> BufferInfo {
        let context = &self.context;
        *self
            .buffer_info
            .entry(BufferInfoKey {
                desc: *desc,
                all_usage_flags,
            })
            .or_insert_with(|| {
                let buffer = context.device.create_buffer_from_desc(&desc, all_usage_flags).unwrap();
                let mem_req = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                unsafe { context.device.destroy_buffer(Some(buffer), None) };
                BufferInfo { mem_req }
            })
    }

    pub fn get_buffer(
        &mut self,
        desc: &BufferDesc,
        info: &BufferInfo,
        alloc: &Alloc,
        all_usage_flags: vk::BufferUsageFlags,
    ) -> UniqueBuffer {
        let context = &self.context;
        *self
            .buffer
            .entry(BufferKey {
                desc: *desc,
                all_usage_flags,
                alloc: *alloc,
            })
            .or_insert_with(|| {
                let buffer = context.device.create_buffer_from_desc(&desc, all_usage_flags).unwrap();

                let mem_req_check = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                assert_eq!(info.mem_req, mem_req_check);

                unsafe { context.device.bind_buffer_memory(buffer, alloc.mem, alloc.offset) }.unwrap();

                Unique::new(buffer, context.allocate_handle_uid())
            })
    }

    pub fn get_image_info(&mut self, desc: &ImageDesc, all_usage_flags: vk::ImageUsageFlags) -> ImageInfo {
        let context = &self.context;
        *self
            .image_info
            .entry(ImageInfoKey {
                desc: *desc,
                all_usage_flags,
            })
            .or_insert_with(|| {
                let image = context.device.create_image_from_desc(&desc, all_usage_flags).unwrap();
                let mem_req = unsafe { context.device.get_image_memory_requirements(image) };
                unsafe { context.device.destroy_image(Some(image), None) };
                ImageInfo { mem_req }
            })
    }

    pub fn get_image(
        &mut self,
        desc: &ImageDesc,
        info: &ImageInfo,
        alloc: &Alloc,
        all_usage_flags: vk::ImageUsageFlags,
    ) -> UniqueImage {
        let context = &self.context;
        *self
            .image
            .entry(ImageKey {
                desc: *desc,
                all_usage_flags,
                alloc: *alloc,
            })
            .or_insert_with(|| {
                let image = context.device.create_image_from_desc(&desc, all_usage_flags).unwrap();

                let mem_req_check = unsafe { context.device.get_image_memory_requirements(image) };
                assert_eq!(info.mem_req, mem_req_check);

                unsafe { context.device.bind_image_memory(image, alloc.mem, alloc.offset) }.unwrap();

                Unique::new(image, context.allocate_handle_uid())
            })
    }

    pub fn get_image_view(&mut self, desc: &ImageDesc, image: UniqueImage) -> UniqueImageView {
        let context = &self.context;
        *self.image_view.entry(ImageViewKey { image }).or_insert_with(|| {
            let image_view_create_info = vk::ImageViewCreateInfo {
                image: Some(image.0),
                view_type: desc.view_type(),
                format: desc.format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: desc.aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            };
            let image_view = unsafe { context.device.create_image_view(&image_view_create_info, None) }.unwrap();
            Unique::new(image_view, context.allocate_handle_uid())
        })
    }
}

impl RenderCache {
    pub const MAX_ATTACHMENTS: usize = 3;

    pub fn new(context: &Arc<Context>) -> Self {
        Self {
            context: Arc::clone(&context),
            render_pass: HashMap::new(),
            framebuffer: HashMap::new(),
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        for (name, len) in &[
            ("render pass", self.render_pass.len()),
            ("framebuffer", self.framebuffer.len()),
        ] {
            ui.text(name);
            ui.next_column();
            ui.text(format!("{}", len));
            ui.next_column();
        }
    }

    pub fn get_render_pass(
        &mut self,
        color_format: vk::Format,
        depth_format: Option<vk::Format>,
        samples: vk::SampleCountFlags,
    ) -> UniqueRenderPass {
        let context = &self.context;
        *self
            .render_pass
            .entry(RenderPassKey {
                color_format,
                depth_format,
                samples,
            })
            .or_insert_with(|| {
                let render_pass = {
                    let is_msaa = samples != vk::SampleCountFlags::N1;
                    let mut attachments = ArrayVec::<[_; Self::MAX_ATTACHMENTS]>::new();
                    let subpass_color_attachment = {
                        let index = attachments.len() as u32;
                        attachments.push(vk::AttachmentDescription {
                            flags: vk::AttachmentDescriptionFlags::empty(),
                            format: color_format,
                            samples,
                            load_op: vk::AttachmentLoadOp::CLEAR,
                            store_op: if is_msaa {
                                vk::AttachmentStoreOp::DONT_CARE
                            } else {
                                vk::AttachmentStoreOp::STORE
                            },
                            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        });
                        vk::AttachmentReference {
                            attachment: index,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }
                    };
                    let subpass_resolve_attachment = if is_msaa {
                        let index = attachments.len() as u32;
                        attachments.push(vk::AttachmentDescription {
                            flags: vk::AttachmentDescriptionFlags::empty(),
                            format: color_format,
                            samples: vk::SampleCountFlags::N1,
                            load_op: vk::AttachmentLoadOp::DONT_CARE,
                            store_op: vk::AttachmentStoreOp::STORE,
                            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        });
                        Some(vk::AttachmentReference {
                            attachment: index,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        })
                    } else {
                        None
                    };
                    let subpass_depth_attachment = if let Some(depth_format) = depth_format {
                        let index = attachments.len() as u32;
                        attachments.push(vk::AttachmentDescription {
                            flags: vk::AttachmentDescriptionFlags::empty(),
                            format: depth_format,
                            samples,
                            load_op: vk::AttachmentLoadOp::CLEAR,
                            store_op: vk::AttachmentStoreOp::DONT_CARE,
                            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        });
                        Some(vk::AttachmentReference {
                            attachment: index,
                            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        })
                    } else {
                        None
                    };
                    let subpass_description = vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .p_color_attachments(
                            slice::from_ref(&subpass_color_attachment),
                            subpass_resolve_attachment.as_ref().map(slice::from_ref),
                        )
                        .p_depth_stencil_attachment(subpass_depth_attachment.as_ref());
                    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
                        .p_attachments(&attachments)
                        .p_subpasses(slice::from_ref(&subpass_description));
                    unsafe { context.device.create_render_pass(&render_pass_create_info, None) }.unwrap()
                };
                Unique::new(render_pass, context.allocate_handle_uid())
            })
    }

    pub fn get_framebuffer(
        &mut self,
        render_pass: UniqueRenderPass,
        size: UVec2,
        color_output_image_view: UniqueImageView,
        color_temp_image_view: Option<UniqueImageView>,
        depth_temp_image_view: Option<UniqueImageView>,
    ) -> UniqueFramebuffer {
        let context = &self.context;
        *self
            .framebuffer
            .entry(FramebufferKey {
                render_pass,
                size,
                color_output_image_view,
                color_temp_image_view,
                depth_temp_image_view,
            })
            .or_insert_with(|| {
                let mut attachments = ArrayVec::<[_; Self::MAX_ATTACHMENTS]>::new();
                if let Some(color_temp_image_view) = color_temp_image_view {
                    attachments.push(color_temp_image_view.0);
                }
                attachments.push(color_output_image_view.0);
                if let Some(depth_temp_image_view) = depth_temp_image_view {
                    attachments.push(depth_temp_image_view.0);
                }

                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass.0)
                    .p_attachments(&attachments)
                    .width(size.x)
                    .height(size.y)
                    .layers(1);
                let framebuffer = unsafe { context.device.create_framebuffer(&framebuffer_create_info, None) }.unwrap();
                Unique::new(framebuffer, context.allocate_handle_uid())
            })
    }
}

impl Drop for RenderCache {
    fn drop(&mut self) {
        for (_, framebuffer) in self.framebuffer.drain() {
            unsafe {
                self.context.device.destroy_framebuffer(Some(framebuffer.0), None);
            }
        }
        for (_, render_pass) in self.render_pass.drain() {
            unsafe {
                self.context.device.destroy_render_pass(Some(render_pass.0), None);
            }
        }
    }
}

impl Drop for ResourceCache {
    fn drop(&mut self) {
        for (_, image_view) in self.image_view.drain() {
            unsafe {
                self.context.device.destroy_image_view(Some(image_view.0), None);
            }
        }
        for (_, image) in self.image.drain() {
            unsafe {
                self.context.device.destroy_image(Some(image.0), None);
            }
        }
        for (_, buffer) in self.buffer.drain() {
            unsafe {
                self.context.device.destroy_buffer(Some(buffer.0), None);
            }
        }
    }
}
