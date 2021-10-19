use crate::prelude::*;
use arrayvec::ArrayVec;
use imgui::Ui;
use spark::{vk, Builder, Device};
use std::{collections::HashMap, iter, slice};
use tinyvec::ArrayVec as TinyVec;

pub trait FormatExt {
    fn bits_per_element(&self) -> usize;
}

impl FormatExt for vk::Format {
    fn bits_per_element(&self) -> usize {
        match *self {
            vk::Format::BC1_RGB_SRGB_BLOCK => 4,
            vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::R16G16_UNORM
            | vk::Format::R16G16_UINT
            | vk::Format::R32_SFLOAT => 32,
            vk::Format::R16G16B16A16_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32G32_UINT => 64,
            vk::Format::R32G32B32A32_SFLOAT | vk::Format::R32G32B32A32_UINT => 128,
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
    pub width: u32,
    pub height_or_zero: u32,
    pub layer_count_or_zero: u32,
    pub mip_count: u32,
    pub formats: TinyVec<[vk::Format; 2]>,
    pub aspect_mask: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,
}

impl ImageDesc {
    pub fn new_1d(width: u32, format: vk::Format, aspect_mask: vk::ImageAspectFlags) -> Self {
        ImageDesc {
            width,
            height_or_zero: 0,
            layer_count_or_zero: 0,
            mip_count: 1,
            formats: iter::once(format).collect(),
            aspect_mask,
            samples: vk::SampleCountFlags::N1,
        }
    }

    pub fn new_2d(size: UVec2, format: vk::Format, aspect_mask: vk::ImageAspectFlags) -> Self {
        ImageDesc {
            width: size.x,
            height_or_zero: size.y,
            layer_count_or_zero: 0,
            mip_count: 1,
            formats: iter::once(format).collect(),
            aspect_mask,
            samples: vk::SampleCountFlags::N1,
        }
    }

    pub fn with_additional_format(mut self, format: vk::Format) -> Self {
        self.formats.push(format);
        self
    }

    pub fn with_layer_count(mut self, layer_count: u32) -> Self {
        self.layer_count_or_zero = layer_count;
        self
    }

    pub fn with_mip_count(mut self, mip_count: u32) -> Self {
        self.mip_count = mip_count;
        self
    }

    pub fn with_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    pub fn first_format(&self) -> vk::Format {
        *self.formats.first().unwrap()
    }

    pub fn size(&self) -> UVec2 {
        UVec2::new(self.width, self.height_or_zero.max(1))
    }

    pub fn extent_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height_or_zero.max(1),
        }
    }

    pub fn is_array(&self) -> bool {
        self.layer_count_or_zero != 1
    }

    pub(crate) fn image_type(&self) -> vk::ImageType {
        if self.height_or_zero == 0 {
            vk::ImageType::N1D
        } else {
            vk::ImageType::N2D
        }
    }

    pub(crate) fn image_view_type(&self) -> vk::ImageViewType {
        if self.height_or_zero == 0 {
            if self.layer_count_or_zero == 0 {
                vk::ImageViewType::N1D
            } else {
                vk::ImageViewType::N1D_ARRAY
            }
        } else {
            if self.layer_count_or_zero == 0 {
                vk::ImageViewType::N2D
            } else {
                vk::ImageViewType::N2D_ARRAY
            }
        }
    }

    pub(crate) fn staging_size(&self) -> usize {
        assert_eq!(self.samples, vk::SampleCountFlags::N1);
        let bits_per_element = self.first_format().bits_per_element();
        let layer_count = self.layer_count_or_zero.max(1) as usize;
        let mut mip_width = self.width as usize;
        let mut mip_height = self.height_or_zero.max(1) as usize;
        let mut mip_offset = 0;
        for _ in 0..self.mip_count {
            let mip_layer_size = (mip_width * mip_height * bits_per_element) / 8;
            mip_offset += mip_layer_size * layer_count;
            mip_width /= 2;
            mip_height /= 2;
        }
        mip_offset
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageViewDesc {
    pub format: Option<vk::Format>,
}

impl ImageViewDesc {
    pub fn with_format(mut self, format: vk::Format) -> Self {
        self.format = Some(format);
        self
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
        let mut format_list = vk::ImageFormatListCreateInfo::builder().p_view_formats(&desc.formats);
        let image_create_info = vk::ImageCreateInfo::builder()
            .flags(if desc.formats.len() > 1 {
                vk::ImageCreateFlags::MUTABLE_FORMAT
            } else {
                vk::ImageCreateFlags::empty()
            })
            .image_type(desc.image_type())
            .format(desc.first_format())
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height_or_zero.max(1),
                depth: 1,
            })
            .mip_levels(desc.mip_count)
            .array_layers(desc.layer_count_or_zero.max(1))
            .samples(desc.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage_flags)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .insert_next(&mut format_list);
        unsafe { self.create_image(&image_create_info, None) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RenderPassDepth {
    pub(crate) format: vk::Format,
    pub(crate) load_op: AttachmentLoadOp,
    pub(crate) store_op: AttachmentStoreOp,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RenderPassKey {
    color_format: Option<vk::Format>,
    depth: Option<RenderPassDepth>,
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
enum BufferAccelLevel {
    Bottom,
    Top,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BufferAccelKey {
    buffer: UniqueBuffer,
    level: BufferAccelLevel,
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
    view_desc: ImageViewDesc,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FramebufferKey {
    render_pass: UniqueRenderPass,
    size: UVec2,
    color_output_image_view: Option<UniqueImageView>,
    color_temp_image_view: Option<UniqueImageView>,
    depth_image_view: Option<UniqueImageView>,
}

pub(crate) struct ResourceCache {
    context: SharedContext,
    buffer_info: HashMap<BufferInfoKey, BufferInfo>,
    buffer: HashMap<BufferKey, UniqueBuffer>,
    buffer_accel: HashMap<BufferAccelKey, vk::AccelerationStructureKHR>,
    image_info: HashMap<ImageInfoKey, ImageInfo>,
    image: HashMap<ImageKey, UniqueImage>,
    image_view: HashMap<ImageViewKey, UniqueImageView>,
}

pub(crate) struct RenderCache {
    context: SharedContext,
    render_pass: HashMap<RenderPassKey, UniqueRenderPass>,
    framebuffer: HashMap<FramebufferKey, UniqueFramebuffer>,
}

impl ResourceCache {
    pub fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            buffer_info: HashMap::new(),
            buffer: HashMap::new(),
            buffer_accel: HashMap::new(),
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
                let buffer = context.device.create_buffer_from_desc(desc, all_usage_flags).unwrap();
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
                let buffer = context.device.create_buffer_from_desc(desc, all_usage_flags).unwrap();

                let mem_req_check = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                assert_eq!(info.mem_req, mem_req_check);

                unsafe { context.device.bind_buffer_memory(buffer, alloc.mem, alloc.offset) }.unwrap();

                Unique::new(buffer, context.allocate_handle_uid())
            })
    }

    pub fn get_buffer_accel(
        &mut self,
        desc: &BufferDesc,
        buffer: UniqueBuffer,
        all_usage: BufferUsage,
    ) -> Option<vk::AccelerationStructureKHR> {
        let level = if all_usage.contains(BufferUsage::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_WRITE) {
            Some(BufferAccelLevel::Bottom)
        } else if all_usage.contains(BufferUsage::TOP_LEVEL_ACCELERATION_STRUCTURE_WRITE) {
            Some(BufferAccelLevel::Top)
        } else {
            None
        };
        let key = BufferAccelKey { buffer, level: level? };
        let context = &self.context;
        Some(*self.buffer_accel.entry(key).or_insert_with(|| {
            let create_info = vk::AccelerationStructureCreateInfoKHR {
                buffer: Some(buffer.0),
                size: desc.size as vk::DeviceSize,
                ty: match key.level {
                    BufferAccelLevel::Bottom => vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    BufferAccelLevel::Top => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                },
                ..Default::default()
            };
            unsafe { context.device.create_acceleration_structure_khr(&create_info, None) }.unwrap()
        }))
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
                let image = context.device.create_image_from_desc(desc, all_usage_flags).unwrap();
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
                let image = context.device.create_image_from_desc(desc, all_usage_flags).unwrap();

                let mem_req_check = unsafe { context.device.get_image_memory_requirements(image) };
                assert_eq!(info.mem_req, mem_req_check);

                unsafe { context.device.bind_image_memory(image, alloc.mem, alloc.offset) }.unwrap();

                Unique::new(image, context.allocate_handle_uid())
            })
    }

    pub fn get_image_view(
        &mut self,
        desc: &ImageDesc,
        image: UniqueImage,
        view_desc: ImageViewDesc,
    ) -> UniqueImageView {
        let context = &self.context;
        *self
            .image_view
            .entry(ImageViewKey { image, view_desc })
            .or_insert_with(|| {
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image: Some(image.0),
                    view_type: desc.image_view_type(),
                    format: view_desc.format.unwrap_or(desc.first_format()),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: desc.aspect_mask,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
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

    pub fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
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
        color_format: Option<vk::Format>,
        depth: Option<RenderPassDepth>,
        samples: vk::SampleCountFlags,
    ) -> UniqueRenderPass {
        let context = &self.context;
        *self
            .render_pass
            .entry(RenderPassKey {
                color_format,
                depth,
                samples,
            })
            .or_insert_with(|| {
                let render_pass = {
                    let is_msaa = samples != vk::SampleCountFlags::N1;
                    let mut attachments = ArrayVec::<_, { Self::MAX_ATTACHMENTS }>::new();
                    let subpass_color_attachment = color_format.map(|color_format| {
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
                    });
                    let subpass_resolve_attachment = color_format.filter(|_| is_msaa).map(|color_format| {
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
                        vk::AttachmentReference {
                            attachment: index,
                            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        }
                    });
                    let subpass_depth_attachment = depth.as_ref().map(|depth| {
                        let index = attachments.len() as u32;
                        attachments.push(vk::AttachmentDescription {
                            flags: vk::AttachmentDescriptionFlags::empty(),
                            format: depth.format,
                            samples,
                            load_op: match depth.load_op {
                                AttachmentLoadOp::Load => vk::AttachmentLoadOp::LOAD,
                                AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                            },
                            store_op: match depth.store_op {
                                AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
                                AttachmentStoreOp::None => vk::AttachmentStoreOp::DONT_CARE,
                            },
                            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                            initial_layout: match depth.load_op {
                                AttachmentLoadOp::Load => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                AttachmentLoadOp::Clear => vk::ImageLayout::UNDEFINED,
                            },
                            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        });
                        vk::AttachmentReference {
                            attachment: index,
                            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        }
                    });
                    let subpass_description = vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .p_color_attachments(
                            subpass_color_attachment.as_ref().map(slice::from_ref).unwrap_or(&[]),
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
        color_output_image_view: Option<UniqueImageView>,
        color_temp_image_view: Option<UniqueImageView>,
        depth_image_view: Option<UniqueImageView>,
    ) -> UniqueFramebuffer {
        let context = &self.context;
        *self
            .framebuffer
            .entry(FramebufferKey {
                render_pass,
                size,
                color_output_image_view,
                color_temp_image_view,
                depth_image_view,
            })
            .or_insert_with(|| {
                let mut attachments = ArrayVec::<_, { Self::MAX_ATTACHMENTS }>::new();
                if let Some(image_view) = color_temp_image_view {
                    attachments.push(image_view.0);
                }
                if let Some(image_view) = color_output_image_view {
                    attachments.push(image_view.0);
                }
                if let Some(image_view) = depth_image_view {
                    attachments.push(image_view.0);
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
        for (_, accel) in self.buffer_accel.drain() {
            unsafe {
                self.context
                    .device
                    .destroy_acceleration_structure_khr(Some(accel), None);
            }
        }
        for (_, buffer) in self.buffer.drain() {
            unsafe {
                self.context.device.destroy_buffer(Some(buffer.0), None);
            }
        }
    }
}
