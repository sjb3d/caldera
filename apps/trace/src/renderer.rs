use crate::accel::*;
use crate::scene::*;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::*;
use imgui::{im_str, Slider, StyleColor, Ui};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use spark::{vk, Builder};
use std::{
    collections::HashMap,
    fs::File,
    ops::{BitOr, BitOrAssign},
    path::PathBuf,
    sync::Arc,
};
use std::{mem, slice};

trait UnitScale {
    fn unit_scale(&self, world_from_local: Similarity3) -> f32;
}

impl UnitScale for Geometry {
    fn unit_scale(&self, world_from_local: Similarity3) -> f32 {
        let local_offset = match self {
            Geometry::TriangleMesh { min, max, .. } => max.abs().max_by_component(min.abs()).component_max(),
            Geometry::Quad { local_from_quad, size } => {
                local_from_quad.translation.abs().component_max() + local_from_quad.scale * size.abs().component_max()
            }
            Geometry::Sphere { centre, radius } => centre.abs().component_max() + radius,
        };
        let world_offset = world_from_local.translation.abs().component_max();
        world_offset.max(world_from_local.scale.abs() * local_offset)
    }
}

trait LightCanBeSampled {
    fn can_be_sampled(&self) -> bool;
}

impl LightCanBeSampled for Light {
    fn can_be_sampled(&self) -> bool {
        match self {
            Light::Dome { .. } => false,
            Light::SolidAngle { .. } => true,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Contiguous, PartialEq, Eq)]
enum LightType {
    Quad,
    Sphere,
    Dome,
    SolidAngle,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct LightInfoEntry {
    light_type: u32,
    probability: f32,
    params_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct QuadLightParams {
    emission: Vec3,
    unit_scale: f32,
    area_pdf: f32,
    normal_ws: Vec3,
    corner_ws: Vec3,
    edge0_ws: Vec3,
    edge1_ws: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SphereLightParams {
    emission: Vec3,
    unit_scale: f32,
    centre_ws: Vec3,
    radius_ws: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct DomeLightParams {
    emission: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SolidAngleLightParams {
    emission: Vec3,
    direction_ws: Vec3,
    solid_angle: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, Default)]
struct PathTraceFlags(u32);

impl PathTraceFlags {
    const ACCUMULATE_ROUGHNESS: PathTraceFlags = PathTraceFlags(0x1);
    const ALLOW_LIGHT_SAMPLING: PathTraceFlags = PathTraceFlags(0x2);
    const ALLOW_BSDF_SAMPLING: PathTraceFlags = PathTraceFlags(0x4);
    const SPHERE_LIGHT_SAMPLE_SOLID_ANGLE: PathTraceFlags = PathTraceFlags(0x8);
    const QUAD_LIGHT_IS_TWO_SIDED: PathTraceFlags = PathTraceFlags(0x10);
}

impl BitOr for PathTraceFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for PathTraceFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Contiguous, PartialEq, Eq)]
pub enum MultipleImportanceHeuristic {
    None,
    Balance,
    Power2,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PathTraceUniforms {
    light_info_table_address: u64,
    light_alias_table_address: u64,
    light_params_base_address: u64,
    sampled_light_count: u32,
    external_light_begin: u32,
    external_light_end: u32,
    world_from_camera: Transform3,
    fov_size_at_unit_z: Vec2,
    sample_index: u32,
    max_segment_count: u32,
    render_color_space: u32,
    mis_heuristic: u32,
    flags: PathTraceFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct LightAliasEntry {
    split: f32,
    indices: u32,
}

descriptor_set_layout!(PathTraceDescriptorSetLayout {
    path_trace_uniforms: UniformData<PathTraceUniforms>,
    accel: AccelerationStructure,
    samples: StorageImage,
    result: [StorageImage; 3],
});

#[repr(transparent)]
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct TextureIndex(u16);

#[derive(Clone, Copy, Default)]
struct ShaderData {
    reflectance_texture: Option<TextureIndex>,
}

#[derive(Clone, Copy, Default)]
struct GeometryAttribData {
    normal_buffer: Option<StaticBufferHandle>,
    uv_buffer: Option<StaticBufferHandle>,
}

#[derive(Clone, Copy)]
enum GeometryRecordData {
    Triangles {
        index_buffer_address: u64,
        position_buffer_address: u64,
        normal_buffer_address: u64,
        uv_buffer_address: u64,
    },
    Sphere,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SamplingTechnique {
    LightsOnly,
    SurfacesOnly,
    LightsAndSurfaces,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Contiguous)]
enum BsdfType {
    None,
    Diffuse,
    Mirror,
    SmoothDielectric,
    SmoothPlastic,
    RoughPlastic,
    RoughConductor,
}

impl From<Surface> for BsdfType {
    fn from(surface: Surface) -> Self {
        match surface {
            Surface::None => BsdfType::None,
            Surface::Diffuse => BsdfType::Diffuse,
            Surface::Mirror => BsdfType::Mirror,
            Surface::SmoothDielectric => BsdfType::SmoothDielectric,
            Surface::SmoothPlastic => BsdfType::SmoothPlastic,
            Surface::RoughPlastic { .. } => BsdfType::RoughPlastic,
            Surface::RoughConductor { .. } => BsdfType::RoughConductor,
        }
    }
}

const MIN_ROUGHNESS: f32 = 0.03;

trait Roughness {
    fn roughness(&self) -> f32;
}

impl Roughness for Surface {
    fn roughness(&self) -> f32 {
        match self {
            Surface::None | Surface::Diffuse => 1.0,
            Surface::Mirror | Surface::SmoothDielectric | Surface::SmoothPlastic => 0.0,
            Surface::RoughPlastic { roughness } | Surface::RoughConductor { roughness } => roughness.max(MIN_ROUGHNESS),
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Zeroable, Pod, Default)]
struct ExtendShaderFlags(u32);

impl ExtendShaderFlags {
    const HAS_NORMALS: ExtendShaderFlags = ExtendShaderFlags(0x0010_0000);
    const HAS_TEXTURE: ExtendShaderFlags = ExtendShaderFlags(0x0020_0000);
    const IS_EMISSIVE: ExtendShaderFlags = ExtendShaderFlags(0x0040_0000);

    fn new(bsdf_type: BsdfType, texture_index: Option<TextureIndex>) -> Self {
        let mut flags = Self(bsdf_type.into_integer() << 16);
        if let Some(texture_index) = texture_index {
            flags |= Self(texture_index.0 as u32) | Self::HAS_TEXTURE;
        }
        flags
    }
}

impl BitOr for ExtendShaderFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for ExtendShaderFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, Default)]
struct ExtendShader {
    flags: ExtendShaderFlags,
    reflectance: Vec3,
    roughness: f32,
    light_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendTriangleHitRecord {
    index_buffer_address: u64,
    position_buffer_address: u64,
    normal_buffer_address: u64,
    uv_buffer_address: u64,
    unit_scale: f32,
    shader: ExtendShader,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SphereGeomData {
    centre: Vec3,
    radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendSphereHitRecord {
    geom: SphereGeomData,
    unit_scale: f32,
    shader: ExtendShader,
}

#[repr(usize)]
#[derive(Clone, Copy, Contiguous)]
enum ShaderGroup {
    RayGenerator,
    ExtendMiss,
    ExtendHitTriangle,
    ExtendHitSphere,
    OcclusionMiss,
    OcclusionHitTriangle,
    OcclusionHitSphere,
}

#[derive(Clone, Copy)]
struct ShaderBindingRegion {
    offset: u32,
    stride: u32,
    size: u32,
}

impl ShaderBindingRegion {
    fn into_device_address_region(self, base_device_address: vk::DeviceAddress) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_device_address + self.offset as vk::DeviceSize,
            stride: self.stride as vk::DeviceSize,
            size: self.size as vk::DeviceSize,
        }
    }
}

struct ShaderBindingData {
    raygen_region: ShaderBindingRegion,
    miss_region: ShaderBindingRegion,
    hit_region: ShaderBindingRegion,
    shader_binding_table: StaticBufferHandle,
    light_info_table: StaticBufferHandle,
    sampled_light_count: u32,
    external_light_begin: u32,
    external_light_end: u32,
}

struct TextureBindingSet {
    context: Arc<Context>,
    sampler: vk::Sampler,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    images: Vec<StaticImageHandle>,
    is_written: bool,
}

impl TextureBindingSet {
    fn new(context: &Arc<Context>, images: Vec<StaticImageHandle>) -> Self {
        // create a separate descriptor pool for bindless textures
        let sampler = {
            let create_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                ..Default::default()
            };
            unsafe { context.device.create_sampler(&create_info, None) }.unwrap()
        };
        let descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: images.len() as u32,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }];
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().p_bindings(&bindings);
            unsafe { context.device.create_descriptor_set_layout(&create_info, None) }.unwrap()
        };
        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (images.len() as u32).max(1),
            }];
            let create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .p_pool_sizes(&pool_sizes);
            unsafe { context.device.create_descriptor_pool(&create_info, None) }.unwrap()
        };
        let descriptor_set = {
            let variable_count = images.len() as u32;
            let mut variable_count_allocate_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                .p_descriptor_counts(slice::from_ref(&variable_count));

            let allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .p_set_layouts(slice::from_ref(&descriptor_set_layout))
                .insert_next(&mut variable_count_allocate_info);

            unsafe { context.device.allocate_descriptor_sets_single(&allocate_info) }.unwrap()
        };

        Self {
            context: Arc::clone(context),
            sampler,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            images,
            is_written: false,
        }
    }

    fn try_write(&mut self, resource_loader: &ResourceLoader) -> Option<()> {
        let mut image_info = Vec::new();
        for image in self.images.iter().cloned() {
            image_info.push(vk::DescriptorImageInfo {
                sampler: Some(self.sampler),
                image_view: Some(resource_loader.get_image_view(image)?),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });
        }
        if !image_info.is_empty() {
            let write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .p_image_info(&image_info)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

            unsafe { self.context.device.update_descriptor_sets(slice::from_ref(&write), &[]) };
        }
        Some(())
    }

    fn update(&mut self, resource_loader: &ResourceLoader) {
        if !self.is_written {
            if self.try_write(resource_loader).is_some() {
                self.is_written = true;
            }
        }
    }

    fn descriptor_set(&self) -> Option<vk::DescriptorSet> {
        if self.is_written {
            Some(self.descriptor_set)
        } else {
            None
        }
    }
}

impl Drop for TextureBindingSet {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_pool(Some(self.descriptor_pool), None);
            self.context
                .device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
            self.context.device.destroy_sampler(Some(self.sampler), None);
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Contiguous, Eq, PartialEq)]
pub enum RenderColorSpace {
    Rec709 = 0,
    ACEScg = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Contiguous)]
pub enum FilterType {
    Box,
    Gaussian,
    Mitchell,
}

pub struct RendererParams {
    pub size: UVec2,
    pub max_bounces: u32,
    pub accumulate_roughness: bool,
    pub sphere_light_sample_solid_angle: bool,
    pub quad_light_is_two_sided: bool,
    pub sampling_technique: SamplingTechnique,
    pub mis_heuristic: MultipleImportanceHeuristic,
    pub render_color_space: RenderColorSpace,
    pub filter_type: FilterType,
}

impl Default for RendererParams {
    fn default() -> Self {
        Self {
            size: UVec2::new(1920, 1080),
            max_bounces: 8,
            accumulate_roughness: true,
            sphere_light_sample_solid_angle: true,
            quad_light_is_two_sided: false,
            sampling_technique: SamplingTechnique::LightsAndSurfaces,
            mis_heuristic: MultipleImportanceHeuristic::Balance,
            render_color_space: RenderColorSpace::ACEScg,
            filter_type: FilterType::Gaussian,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SamplePixel {
    x: u16,
    y: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct FilterData {
    image_size: UVec2,
    sample_index: u32,
    filter_type: u32,
}

descriptor_set_layout!(FilterDescriptorSetLayout {
    data: UniformData<FilterData>,
    samples: StorageImage,
    input: [StorageImage; 3],
    result: StorageImage,
});

pub struct RenderProgress {
    next_sample_index: u32,
}

impl RenderProgress {
    pub fn new() -> Self {
        Self { next_sample_index: 0 }
    }

    pub fn reset(&mut self) {
        self.next_sample_index = 0;
    }

    pub fn done(&self) -> bool {
        self.next_sample_index == Renderer::SAMPLES_PER_SEQUENCE
    }
}

pub struct Renderer {
    context: Arc<Context>,
    scene: Arc<Scene>,
    accel: SceneAccel,

    path_trace_pipeline_layout: vk::PipelineLayout,
    path_trace_descriptor_set_layout: PathTraceDescriptorSetLayout,
    path_trace_pipeline: vk::Pipeline,

    filter_descriptor_set_layout: FilterDescriptorSetLayout,
    filter_pipeline_layout: vk::PipelineLayout,

    texture_binding_set: TextureBindingSet,
    shader_data: Vec<ShaderData>,
    geometry_attrib_data: Vec<GeometryAttribData>,
    shader_binding_data: Option<ShaderBindingData>,

    sample_image: StaticImageHandle,
    result_image: ImageHandle,

    pub params: RendererParams,
}

impl Renderer {
    const SEQUENCE_COUNT: u32 = 1024;
    const SAMPLES_PER_SEQUENCE: u32 = 256;

    const HIT_ENTRY_COUNT_PER_INSTANCE: u32 = 2;
    const MISS_ENTRY_COUNT: u32 = 2;

    pub fn new(
        context: &Arc<Context>,
        scene: &Arc<Scene>,
        descriptor_set_layout_cache: &mut DescriptorSetLayoutCache,
        pipeline_cache: &PipelineCache,
        resource_loader: &mut ResourceLoader,
        render_graph: &mut RenderGraph,
        global_allocator: &mut Allocator,
        params: RendererParams,
    ) -> Self {
        let accel = SceneAccel::new(context, scene, resource_loader);

        // start loading textures and attributes for all meshes
        let mut texture_indices = HashMap::<PathBuf, TextureIndex>::new();
        let mut texture_images = Vec::new();
        let mut shader_data = vec![ShaderData::default(); scene.materials.len()];
        let mut geometry_attrib_data = vec![GeometryAttribData::default(); scene.geometries.len()];
        for instance_ref in accel.clusters().instance_iter().cloned() {
            let instance = scene.instance(instance_ref);
            let material_ref = instance.material_ref;
            let geometry_ref = instance.geometry_ref;
            let material = scene.material(material_ref);
            let geometry = scene.geometry(geometry_ref);

            let has_normals = matches!(geometry, Geometry::TriangleMesh { normals: Some(_), .. });
            let has_uvs_and_texture = matches!(geometry, Geometry::TriangleMesh { uvs: Some(_), .. })
                && matches!(material.reflectance, Reflectance::Texture(_));

            if has_normals {
                geometry_attrib_data
                    .get_mut(geometry_ref.0 as usize)
                    .unwrap()
                    .normal_buffer = {
                    let normal_buffer = resource_loader.create_buffer();
                    resource_loader.async_load({
                        let scene = Arc::clone(scene);
                        move |allocator| {
                            let normals = match scene.geometry(geometry_ref) {
                                Geometry::TriangleMesh {
                                    normals: Some(normals), ..
                                } => normals.as_slice(),
                                _ => unreachable!(),
                            };

                            let buffer_desc = BufferDesc::new(normals.len() * mem::size_of::<Vec3>());
                            let mut mapping = allocator
                                .map_buffer(normal_buffer, &buffer_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                                .unwrap();
                            for n in normals.iter() {
                                mapping.write(n);
                            }
                        }
                    });
                    Some(normal_buffer)
                };
            }

            if has_uvs_and_texture {
                geometry_attrib_data.get_mut(geometry_ref.0 as usize).unwrap().uv_buffer = {
                    let uv_buffer = resource_loader.create_buffer();
                    resource_loader.async_load({
                        let scene = Arc::clone(scene);
                        move |allocator| {
                            let uvs = match scene.geometry(geometry_ref) {
                                Geometry::TriangleMesh { uvs: Some(uvs), .. } => uvs.as_slice(),
                                _ => unreachable!(),
                            };

                            let buffer_desc = BufferDesc::new(uvs.len() * mem::size_of::<Vec2>());
                            let mut mapping = allocator
                                .map_buffer(uv_buffer, &buffer_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                                .unwrap();
                            for uv in uvs.iter() {
                                mapping.write(uv);
                            }
                        }
                    });
                    Some(uv_buffer)
                };

                shader_data
                    .get_mut(material_ref.0 as usize)
                    .unwrap()
                    .reflectance_texture = match &material.reflectance {
                    Reflectance::Texture(filename) => {
                        Some(*texture_indices.entry(filename.clone()).or_insert_with(|| {
                            let image_handle = resource_loader.create_image();
                            resource_loader.async_load({
                                let filename = filename.clone();
                                move |allocator| {
                                    let mut reader = File::open(&filename).unwrap();
                                    let (info, data) =
                                        stb::image::stbi_load_from_reader(&mut reader, stb::image::Channels::RgbAlpha)
                                            .unwrap();
                                    let can_bc_compress = (info.width % 4) == 0 && (info.height % 4) == 0;
                                    let size_in_pixels = UVec2::new(info.width as u32, info.height as u32);
                                    println!(
                                        "loaded {:?}: {}x{} ({})",
                                        filename.file_name().unwrap(),
                                        size_in_pixels.x,
                                        size_in_pixels.y,
                                        if can_bc_compress { "bc1" } else { "rgba" }
                                    );
                                    if can_bc_compress {
                                        // compress on write
                                        let image_desc = ImageDesc::new_2d(
                                            size_in_pixels,
                                            vk::Format::BC1_RGB_SRGB_BLOCK,
                                            vk::ImageAspectFlags::COLOR,
                                        );
                                        let mut mapping = allocator
                                            .map_image(image_handle, &image_desc, ImageUsage::RAY_TRACING_SAMPLED)
                                            .unwrap();
                                        let size_in_blocks = size_in_pixels / 4;
                                        let mut tmp_rgba = [0xffu8; 16 * 4];
                                        let mut dst_bc1 = [0u8; 8];
                                        for block_y in 0..size_in_blocks.y {
                                            for block_x in 0..size_in_blocks.x {
                                                for pixel_y in 0..4 {
                                                    let tmp_offset = (pixel_y * 4 * 4) as usize;
                                                    let tmp_row = &mut tmp_rgba[tmp_offset..(tmp_offset + 16)];
                                                    let src_offset = (((block_y * 4 + pixel_y) * size_in_pixels.x
                                                        + block_x * 4)
                                                        * 4)
                                                        as usize;
                                                    let src_row = &data.as_slice()[src_offset..(src_offset + 16)];
                                                    for pixel_x in 0..4 {
                                                        for component in 0..3 {
                                                            let row_offset = (4 * pixel_x + component) as usize;
                                                            unsafe {
                                                                *tmp_row.get_unchecked_mut(row_offset) =
                                                                    *src_row.get_unchecked(row_offset)
                                                            };
                                                        }
                                                    }
                                                }
                                                stb::dxt::stb_compress_dxt_block(
                                                    &mut dst_bc1,
                                                    &tmp_rgba,
                                                    0,
                                                    stb::dxt::CompressionMode::Normal,
                                                );
                                                mapping.write(&dst_bc1);
                                            }
                                        }
                                    } else {
                                        // load uncompressed
                                        let image_desc = ImageDesc::new_2d(
                                            size_in_pixels,
                                            vk::Format::R8G8B8A8_SRGB,
                                            vk::ImageAspectFlags::COLOR,
                                        );
                                        let mut mapping = allocator
                                            .map_image(image_handle, &image_desc, ImageUsage::RAY_TRACING_SAMPLED)
                                            .unwrap();
                                        mapping.write(data.as_slice());
                                    }
                                }
                            });

                            let texture_index = TextureIndex(texture_images.len() as u16);
                            texture_images.push(image_handle);
                            texture_index
                        }))
                    }
                    _ => unreachable!(),
                };
            }
        }

        let path_trace_descriptor_set_layout = PathTraceDescriptorSetLayout::new(descriptor_set_layout_cache);
        let texture_binding_set = TextureBindingSet::new(context, texture_images);
        let path_trace_pipeline_layout = descriptor_set_layout_cache.create_pipeline_multi_layout(&[
            path_trace_descriptor_set_layout.0,
            texture_binding_set.descriptor_set_layout,
        ]);

        // make pipeline
        let group_desc: Vec<_> = (ShaderGroup::MIN_VALUE..=ShaderGroup::MAX_VALUE)
            .map(|i| match ShaderGroup::from_integer(i).unwrap() {
                ShaderGroup::RayGenerator => RayTracingShaderGroupDesc::Raygen("trace/path_trace.rgen.spv"),
                ShaderGroup::ExtendMiss => RayTracingShaderGroupDesc::Miss("trace/extend.rmiss.spv"),
                ShaderGroup::ExtendHitTriangle => RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/extend_triangle.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
                ShaderGroup::ExtendHitSphere => RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/extend_sphere.rchit.spv",
                    any_hit: None,
                    intersection: Some("trace/sphere.rint.spv"),
                },
                ShaderGroup::OcclusionMiss => RayTracingShaderGroupDesc::Miss("trace/occlusion.rmiss.spv"),
                ShaderGroup::OcclusionHitTriangle => RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/occlusion.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
                ShaderGroup::OcclusionHitSphere => RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/occlusion.rchit.spv",
                    any_hit: None,
                    intersection: Some("trace/sphere.rint.spv"),
                },
            })
            .collect();
        let path_trace_pipeline = pipeline_cache.get_ray_tracing(&group_desc, path_trace_pipeline_layout);

        let filter_descriptor_set_layout = FilterDescriptorSetLayout::new(descriptor_set_layout_cache);
        let filter_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(filter_descriptor_set_layout.0);

        let sample_image = resource_loader.create_image();
        resource_loader.async_load(move |allocator| {
            let sequences: Vec<Vec<_>> = (0..Self::SEQUENCE_COUNT)
                .into_par_iter()
                .map(|i| {
                    let mut rng = SmallRng::seed_from_u64(i as u64);
                    pmj::generate(Self::SAMPLES_PER_SEQUENCE as usize, 4, &mut rng)
                })
                .collect();

            let desc = ImageDesc::new_2d(
                UVec2::new(Self::SAMPLES_PER_SEQUENCE, Self::SEQUENCE_COUNT),
                vk::Format::R16G16_UINT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = allocator
                .map_image(sample_image, &desc, ImageUsage::COMPUTE_STORAGE_READ)
                .unwrap();

            for sample in sequences.iter().flat_map(|sequence| sequence.iter()) {
                let pixel = SamplePixel {
                    x: sample.x_bits(16) as u16,
                    y: sample.y_bits(16) as u16,
                };
                writer.write(&pixel);
            }
        });

        let result_image = {
            let desc = ImageDesc::new_2d(
                params.size,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let usage = ImageUsage::FRAGMENT_STORAGE_READ
                | ImageUsage::COMPUTE_STORAGE_READ
                | ImageUsage::COMPUTE_STORAGE_WRITE;
            render_graph.create_image(&desc, usage, global_allocator)
        };

        Self {
            context: Arc::clone(context),
            scene: Arc::clone(scene),
            accel,
            path_trace_descriptor_set_layout,
            path_trace_pipeline_layout,
            path_trace_pipeline,
            filter_descriptor_set_layout,
            filter_pipeline_layout,
            texture_binding_set,
            shader_data,
            geometry_attrib_data,
            shader_binding_data: None,
            sample_image,
            result_image,
            params,
        }
    }

    fn geometry_attrib_data(&self, geometry_ref: GeometryRef) -> &GeometryAttribData {
        &self.geometry_attrib_data[geometry_ref.0 as usize]
    }

    fn create_shader_binding_table(&self, resource_loader: &mut ResourceLoader) -> Option<ShaderBindingData> {
        // gather the data we need for records
        let mut geometry_records = vec![None; self.scene.geometries.len()];
        for geometry_ref in self.accel.clusters().geometry_iter().cloned() {
            let geometry_buffer_data = self.accel.geometry_accel_data(geometry_ref)?;
            geometry_records[geometry_ref.0 as usize] = match geometry_buffer_data {
                GeometryAccelData::Triangles {
                    index_buffer,
                    position_buffer,
                } => {
                    let index_buffer_address = unsafe {
                        self.context
                            .device
                            .get_buffer_device_address_helper(resource_loader.get_buffer(*index_buffer)?)
                    };
                    let position_buffer_address = unsafe {
                        self.context
                            .device
                            .get_buffer_device_address_helper(resource_loader.get_buffer(*position_buffer)?)
                    };

                    let attrib_data = self.geometry_attrib_data(geometry_ref);
                    let normal_buffer_address = if let Some(buffer) = attrib_data.normal_buffer {
                        unsafe {
                            self.context
                                .device
                                .get_buffer_device_address_helper(resource_loader.get_buffer(buffer)?)
                        }
                    } else {
                        0
                    };
                    let uv_buffer_address = if let Some(buffer) = attrib_data.uv_buffer {
                        unsafe {
                            self.context
                                .device
                                .get_buffer_device_address_helper(resource_loader.get_buffer(buffer)?)
                        }
                    } else {
                        0
                    };
                    Some(GeometryRecordData::Triangles {
                        index_buffer_address,
                        position_buffer_address,
                        normal_buffer_address,
                        uv_buffer_address,
                    })
                }
                GeometryAccelData::Sphere { .. } => Some(GeometryRecordData::Sphere),
            };
        }

        // grab the shader group handles
        let rtpp = self.context.ray_tracing_pipeline_properties.as_ref().unwrap();
        let handle_size = rtpp.shader_group_handle_size as usize;
        let shader_group_count = 1 + ShaderGroup::MAX_VALUE;
        let mut shader_group_handle_data = vec![0u8; shader_group_count * handle_size];
        unsafe {
            self.context.device.get_ray_tracing_shader_group_handles_khr(
                self.path_trace_pipeline,
                0,
                shader_group_count as u32,
                &mut shader_group_handle_data,
            )
        }
        .unwrap();

        // count the number of lights
        let emissive_instance_count = self
            .accel
            .clusters()
            .instance_iter()
            .cloned()
            .filter(|&instance_ref| {
                let material_ref = self.scene.instance(instance_ref).material_ref;
                self.scene.material(material_ref).emission.is_some()
            })
            .count() as u32;
        let external_light_begin = emissive_instance_count;
        let external_light_end = emissive_instance_count + self.scene.lights.len() as u32;
        let sampled_light_count =
            emissive_instance_count + self.scene.lights.iter().filter(|light| light.can_be_sampled()).count() as u32;
        let total_light_count = external_light_end;

        // figure out the layout
        let rtpp = self.context.ray_tracing_pipeline_properties.as_ref().unwrap();

        let align_up = |n: u32, a: u32| (n + a - 1) & !(a - 1);
        let mut next_offset = 0;

        // ray generation
        let raygen_record_size = 0;
        let raygen_stride = align_up(
            rtpp.shader_group_handle_size + raygen_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let raygen_region = ShaderBindingRegion {
            offset: next_offset,
            stride: raygen_stride,
            size: raygen_stride,
        };
        next_offset += align_up(raygen_region.size, rtpp.shader_group_base_alignment);

        // miss shaders
        let miss_record_size = 0;
        let miss_stride = align_up(
            rtpp.shader_group_handle_size + miss_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let miss_entry_count = Self::MISS_ENTRY_COUNT;
        let miss_region = ShaderBindingRegion {
            offset: next_offset,
            stride: miss_stride,
            size: miss_stride * miss_entry_count,
        };
        next_offset += align_up(miss_region.size, rtpp.shader_group_base_alignment);

        // hit shaders
        let hit_record_size =
            mem::size_of::<ExtendTriangleHitRecord>().max(mem::size_of::<ExtendSphereHitRecord>()) as u32;
        let hit_stride = align_up(
            rtpp.shader_group_handle_size + hit_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let hit_entry_count = (self.scene.instances.len() as u32) * Self::HIT_ENTRY_COUNT_PER_INSTANCE;
        let hit_region = ShaderBindingRegion {
            offset: next_offset,
            stride: hit_stride,
            size: hit_stride * hit_entry_count,
        };
        next_offset += align_up(hit_region.size, rtpp.shader_group_base_alignment);

        let total_size = next_offset;

        // write the table
        let shader_binding_table = resource_loader.create_buffer();
        let light_info_table = resource_loader.create_buffer();
        resource_loader.async_load({
            let scene = Arc::clone(&self.scene);
            let clusters = Arc::clone(self.accel.clusters());
            let shader_data = self.shader_data.clone(); // TODO: fix
            move |allocator| {
                let shader_group_handle = |group: ShaderGroup| {
                    let begin = (group.into_integer() as usize) * handle_size;
                    let end = begin + handle_size;
                    &shader_group_handle_data[begin..end]
                };

                let desc = BufferDesc::new(total_size as usize);
                let mut writer = allocator
                    .map_buffer(
                        shader_binding_table,
                        &desc,
                        BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                    )
                    .unwrap();

                assert_eq!(raygen_region.offset, 0);
                writer.write(shader_group_handle(ShaderGroup::RayGenerator));

                writer.write_zeros(miss_region.offset as usize - writer.written());
                {
                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handle(ShaderGroup::ExtendMiss));
                    writer.write_zeros(end_offset - writer.written());

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handle(ShaderGroup::OcclusionMiss));
                    writer.write_zeros(end_offset - writer.written());
                }

                writer.write_zeros(hit_region.offset as usize - writer.written());
                let mut light_info = Vec::new();
                let mut light_params_data = Vec::new();
                let align16 = |n: usize| (n + 15) & !15;
                let align16_vec = |v: &mut Vec<u8>| v.resize(align16(v.len()), 0);
                for instance_ref in clusters.instance_iter().cloned() {
                    let instance = scene.instance(instance_ref);
                    let geometry = scene.geometry(instance.geometry_ref);
                    let transform = scene.transform(instance.transform_ref);
                    let material = scene.material(instance.material_ref);

                    let reflectance_texture = shader_data[instance.material_ref.0 as usize].reflectance_texture;
                    let reflectance = match material.reflectance {
                        Reflectance::Constant(c) => c,
                        Reflectance::Texture(_) => Vec3::zero(),
                    }
                    .clamped(Vec3::zero(), Vec3::one());

                    let mut shader = ExtendShader {
                        flags: ExtendShaderFlags::new(material.surface.into(), reflectance_texture),
                        reflectance,
                        roughness: material.surface.roughness(),
                        light_index: 0,
                    };
                    if let Some(emission) = material.emission {
                        shader.flags |= ExtendShaderFlags::IS_EMISSIVE;
                        shader.light_index = light_info.len() as u32;

                        let world_from_local = transform.world_from_local;
                        let unit_scale = geometry.unit_scale(world_from_local);

                        let params_offset = light_params_data.len();
                        let (light_type, area_ws) = match geometry {
                            Geometry::TriangleMesh { .. } => unimplemented!(),
                            Geometry::Quad { size, local_from_quad } => {
                                let world_from_quad = world_from_local * *local_from_quad;

                                let centre_ws = world_from_quad.translation;
                                let edge0_ws = world_from_quad.transform_vec3(Vec3::new(size.x, 0.0, 0.0));
                                let edge1_ws = world_from_quad.transform_vec3(Vec3::new(0.0, size.y, 0.0));
                                let normal_ws = world_from_quad.transform_vec3(Vec3::unit_z()).normalized();
                                let area_ws = (world_from_quad.scale * world_from_quad.scale * size.x * size.y).abs();

                                light_params_data.extend_from_slice(bytemuck::bytes_of(&QuadLightParams {
                                    emission,
                                    unit_scale,
                                    area_pdf: 1.0 / area_ws,
                                    normal_ws,
                                    corner_ws: centre_ws - 0.5 * (edge0_ws + edge1_ws),
                                    edge0_ws,
                                    edge1_ws,
                                }));

                                (LightType::Quad, area_ws)
                            }
                            Geometry::Sphere { centre, radius } => {
                                let centre_ws = world_from_local * *centre;
                                let radius_ws = (world_from_local.scale * *radius).abs();
                                let area_ws = 4.0 * PI * radius_ws * radius_ws;

                                light_params_data.extend_from_slice(bytemuck::bytes_of(&SphereLightParams {
                                    emission,
                                    unit_scale,
                                    centre_ws,
                                    radius_ws,
                                }));

                                (LightType::Sphere, area_ws)
                            }
                        };

                        align16_vec(&mut light_params_data);
                        light_info.push(LightInfoEntry {
                            light_type: light_type.into_integer(),
                            probability: area_ws * emission.luminance() * PI,
                            params_offset: params_offset as u32,
                        });
                    }

                    let unit_scale = geometry.unit_scale(transform.world_from_local);
                    match geometry_records[instance.geometry_ref.0 as usize].unwrap() {
                        GeometryRecordData::Triangles {
                            index_buffer_address,
                            position_buffer_address,
                            normal_buffer_address,
                            uv_buffer_address,
                        } => {
                            if normal_buffer_address != 0 {
                                shader.flags |= ExtendShaderFlags::HAS_NORMALS;
                            }
                            let hit_record = ExtendTriangleHitRecord {
                                index_buffer_address,
                                position_buffer_address,
                                normal_buffer_address,
                                uv_buffer_address,
                                unit_scale,
                                shader,
                                _pad: 0,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::ExtendHitTriangle));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::OcclusionHitTriangle));
                            writer.write_zeros(end_offset - writer.written());
                        }
                        GeometryRecordData::Sphere => {
                            let geom = match scene.geometry(instance.geometry_ref) {
                                Geometry::Sphere { centre, radius } => SphereGeomData {
                                    centre: *centre,
                                    radius: radius.abs(),
                                },
                                _ => unreachable!(),
                            };
                            let hit_record = ExtendSphereHitRecord {
                                geom,
                                unit_scale,
                                shader,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::ExtendHitSphere));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::OcclusionHitSphere));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());
                        }
                    }
                }

                let local_light_total_power = if light_info.is_empty() {
                    None
                } else {
                    Some(light_info.iter().map(|info| info.probability).sum())
                };

                for light in scene
                    .lights
                    .iter()
                    .filter(|light| light.can_be_sampled())
                    .chain(scene.lights.iter().filter(|light| !light.can_be_sampled()))
                {
                    let params_offset = light_params_data.len();
                    let (light_type, sampling_power) = match light {
                        Light::Dome { emission } => {
                            light_params_data
                                .extend_from_slice(bytemuck::bytes_of(&DomeLightParams { emission: *emission }));

                            (LightType::Dome, 0.0)
                        }
                        Light::SolidAngle {
                            emission,
                            direction_ws,
                            solid_angle,
                        } => {
                            let direction_ws = direction_ws.normalized();
                            let solid_angle = solid_angle.abs();

                            light_params_data.extend_from_slice(bytemuck::bytes_of(&SolidAngleLightParams {
                                emission: *emission,
                                direction_ws,
                                solid_angle: solid_angle.abs(),
                            }));

                            // HACK: use the total local light power to split samples 50/50
                            // TODO: use scene bounds to estimate the light power?
                            let fake_power = local_light_total_power.unwrap_or(1.0);
                            (LightType::SolidAngle, fake_power)
                        }
                    };

                    align16_vec(&mut light_params_data);
                    light_info.push(LightInfoEntry {
                        light_type: light_type.into_integer(),
                        probability: sampling_power,
                        params_offset: params_offset as u32,
                    });
                }
                assert_eq!(light_info.len() as u32, total_light_count);

                let mut next_offset = 0;
                next_offset += align16(light_info.len().max(1) * mem::size_of::<LightInfoEntry>());

                let light_alias_offset = next_offset;
                next_offset += align16((sampled_light_count as usize) * mem::size_of::<LightAliasEntry>());

                let light_params_offset = next_offset;
                next_offset += light_params_data.len();

                let light_info_table_desc = BufferDesc::new(next_offset);
                let mut writer = allocator
                    .map_buffer(
                        light_info_table,
                        &light_info_table_desc,
                        BufferUsage::RAY_TRACING_STORAGE_READ,
                    )
                    .unwrap();

                let rcp_total_sampled_light_power = 1.0 / light_info.iter().map(|info| info.probability).sum::<f32>();
                let mut under = Vec::new();
                let mut over = Vec::new();
                for info in light_info.iter_mut() {
                    info.probability *= rcp_total_sampled_light_power;
                    writer.write(info);
                }

                writer.write_zeros(light_alias_offset - writer.written());
                for (index, probability) in light_info
                    .iter()
                    .map(|info| info.probability)
                    .take(sampled_light_count as usize)
                    .enumerate()
                {
                    let u = probability * (sampled_light_count as f32);
                    if u < 1.0 {
                        under.push((index, u));
                    } else {
                        over.push((index, u));
                    }
                }
                for _ in 0..sampled_light_count {
                    let entry = if let Some((small_i, small_u)) = under.pop() {
                        if let Some((big_i, big_u)) = over.pop() {
                            let remain_u = big_u - (1.0 - small_u);
                            if remain_u < 1.0 {
                                under.push((big_i, remain_u));
                            } else {
                                over.push((big_i, remain_u));
                            }
                            LightAliasEntry {
                                split: small_u,
                                indices: ((big_i << 16) | small_i) as u32,
                            }
                        } else {
                            println!("no alias found for {} (light {})", small_u, small_i);
                            LightAliasEntry {
                                split: 1.0,
                                indices: ((small_i << 16) | small_i) as u32,
                            }
                        }
                    } else {
                        let (big_i, big_u) = over.pop().unwrap();
                        if big_u > 1.0 {
                            println!("no alias found for {} (light {})", big_u, big_i);
                        }
                        LightAliasEntry {
                            split: 1.0,
                            indices: ((big_i << 16) | big_i) as u32,
                        }
                    };
                    writer.write(&entry);
                }

                writer.write_zeros(light_params_offset - writer.written());
                writer.write(light_params_data.as_slice());
            }
        });
        Some(ShaderBindingData {
            raygen_region,
            miss_region,
            hit_region,
            shader_binding_table,
            light_info_table,
            sampled_light_count,
            external_light_begin,
            external_light_end,
        })
    }

    pub fn debug_ui(&mut self, progress: &mut RenderProgress, ui: &Ui) {
        let mut needs_reset = false;

        ui.text("Color Space:");
        needs_reset |= ui.radio_button(
            im_str!("Rec709 (sRGB primaries)"),
            &mut self.params.render_color_space,
            RenderColorSpace::Rec709,
        );
        needs_reset |= ui.radio_button(
            im_str!("ACEScg (AP1 primaries)"),
            &mut self.params.render_color_space,
            RenderColorSpace::ACEScg,
        );

        ui.text("Filter:");
        needs_reset |= ui.radio_button(im_str!("Box"), &mut self.params.filter_type, FilterType::Box);
        needs_reset |= ui.radio_button(im_str!("Gaussian"), &mut self.params.filter_type, FilterType::Gaussian);
        needs_reset |= ui.radio_button(im_str!("Mitchell"), &mut self.params.filter_type, FilterType::Mitchell);

        ui.text("Sampling Technique:");
        needs_reset |= ui.radio_button(
            im_str!("Lights Only"),
            &mut self.params.sampling_technique,
            SamplingTechnique::LightsOnly,
        );
        needs_reset |= ui.radio_button(
            im_str!("Surfaces Only"),
            &mut self.params.sampling_technique,
            SamplingTechnique::SurfacesOnly,
        );
        needs_reset |= ui.radio_button(
            im_str!("Lights And Surfaces"),
            &mut self.params.sampling_technique,
            SamplingTechnique::LightsAndSurfaces,
        );

        let id = ui.push_id(im_str!("MIS Heuristic"));
        if self.params.sampling_technique == SamplingTechnique::LightsAndSurfaces {
            ui.text("MIS Heuristic:");
            needs_reset |= ui.radio_button(
                im_str!("None"),
                &mut self.params.mis_heuristic,
                MultipleImportanceHeuristic::None,
            );
            needs_reset |= ui.radio_button(
                im_str!("Balance"),
                &mut self.params.mis_heuristic,
                MultipleImportanceHeuristic::Balance,
            );
            needs_reset |= ui.radio_button(
                im_str!("Power2"),
                &mut self.params.mis_heuristic,
                MultipleImportanceHeuristic::Power2,
            );
        } else {
            ui.text_disabled("MIS Heuristic:");
            let style = ui.push_style_color(StyleColor::Text, ui.style_color(StyleColor::TextDisabled));
            ui.radio_button_bool(im_str!("None"), true);
            ui.radio_button_bool(im_str!("Balance"), false);
            ui.radio_button_bool(im_str!("Power2"), false);
            style.pop(&ui);
        }
        id.pop(&ui);

        needs_reset |= Slider::new(im_str!("Max Bounces"))
            .range(0..=24)
            .build(&ui, &mut self.params.max_bounces);
        needs_reset |= ui.checkbox(im_str!("Accumulate Roughness"), &mut self.params.accumulate_roughness);
        needs_reset |= ui.checkbox(
            im_str!("Sample Sphere Light Solid Angle"),
            &mut self.params.sphere_light_sample_solid_angle,
        );
        needs_reset |= ui.checkbox(
            im_str!("Quad Light Is Two Sided"),
            &mut self.params.quad_light_is_two_sided,
        );

        if needs_reset {
            progress.reset();
        }
    }

    pub fn update<'a>(
        &mut self,
        context: &'a Context,
        schedule: &mut RenderSchedule<'a>,
        resource_loader: &mut ResourceLoader,
        global_allocator: &mut Allocator,
    ) {
        // continue with acceleration structures
        self.accel.update(
            context,
            resource_loader,
            global_allocator,
            schedule,
            Self::HIT_ENTRY_COUNT_PER_INSTANCE,
        );

        // make the bindless texture set
        self.texture_binding_set.update(resource_loader);

        // make shader binding table
        if self.shader_binding_data.is_none() {
            self.shader_binding_data = self.create_shader_binding_table(resource_loader);
        }
    }

    pub fn render<'a>(
        &'a self,
        progress: &mut RenderProgress,
        context: &'a Context,
        schedule: &mut RenderSchedule<'a>,
        pipeline_cache: &'a PipelineCache,
        descriptor_pool: &'a DescriptorPool,
        resource_loader: &'a ResourceLoader,
        world_from_camera: Similarity3,
        fov_y: f32,
    ) -> Option<ImageHandle> {
        // check readiness
        let top_level_accel = self.accel.top_level_accel()?;
        let texture_binding_descriptor_set = self.texture_binding_set.descriptor_set()?;
        let shader_binding_data = self.shader_binding_data.as_ref()?;
        let shader_binding_table_buffer = resource_loader.get_buffer(shader_binding_data.shader_binding_table)?;
        let light_info_table_buffer = resource_loader.get_buffer(shader_binding_data.light_info_table)?;
        let sample_image_view = resource_loader.get_image_view(self.sample_image)?;

        // do a pass
        if !progress.done() {
            let temp_desc = ImageDesc::new_2d(self.params.size, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR);
            let temp_images = (
                schedule.describe_image(&temp_desc),
                schedule.describe_image(&temp_desc),
                schedule.describe_image(&temp_desc),
            );

            schedule.add_compute(
                command_name!("trace"),
                |params| {
                    self.accel.declare_parameters(params);
                    params.add_image(temp_images.0, ImageUsage::RAY_TRACING_STORAGE_WRITE);
                    params.add_image(temp_images.1, ImageUsage::RAY_TRACING_STORAGE_WRITE);
                    params.add_image(temp_images.2, ImageUsage::RAY_TRACING_STORAGE_WRITE);
                },
                {
                    let sample_index = progress.next_sample_index;
                    move |params, cmd| {
                        let temp_image_views = [
                            params.get_image_view(temp_images.0),
                            params.get_image_view(temp_images.1),
                            params.get_image_view(temp_images.2),
                        ];

                        let aspect_ratio = (self.params.size.x as f32) / (self.params.size.y as f32);
                        let fov_size_at_unit_z = 2.0 * (0.5 * fov_y).tan() * Vec2::new(aspect_ratio, 1.0);

                        let mut path_trace_flags = match self.params.sampling_technique {
                            SamplingTechnique::LightsOnly => PathTraceFlags::ALLOW_LIGHT_SAMPLING,
                            SamplingTechnique::SurfacesOnly => PathTraceFlags::ALLOW_BSDF_SAMPLING,
                            SamplingTechnique::LightsAndSurfaces => {
                                PathTraceFlags::ALLOW_LIGHT_SAMPLING | PathTraceFlags::ALLOW_BSDF_SAMPLING
                            }
                        };
                        if self.params.accumulate_roughness {
                            path_trace_flags |= PathTraceFlags::ACCUMULATE_ROUGHNESS;
                        }
                        if self.params.sphere_light_sample_solid_angle {
                            path_trace_flags |= PathTraceFlags::SPHERE_LIGHT_SAMPLE_SOLID_ANGLE;
                        }
                        if self.params.quad_light_is_two_sided {
                            path_trace_flags |= PathTraceFlags::QUAD_LIGHT_IS_TWO_SIDED;
                        }

                        let light_info_table_address = unsafe {
                            self.context
                                .device
                                .get_buffer_device_address_helper(light_info_table_buffer)
                        };
                        let align16 = |n: u64| (n + 15) & !15;
                        let light_alias_table_address = align16(
                            light_info_table_address
                                + (shader_binding_data.external_light_end as u64)
                                    * (mem::size_of::<LightInfoEntry>() as u64),
                        );
                        let light_params_base_address = align16(
                            light_alias_table_address
                                + (shader_binding_data.sampled_light_count as u64)
                                    * (mem::size_of::<LightAliasEntry>() as u64),
                        );

                        let path_trace_descriptor_set = self.path_trace_descriptor_set_layout.write(
                            descriptor_pool,
                            |buf: &mut PathTraceUniforms| {
                                *buf = PathTraceUniforms {
                                    light_info_table_address,
                                    light_alias_table_address,
                                    light_params_base_address,
                                    sampled_light_count: shader_binding_data.sampled_light_count,
                                    external_light_begin: shader_binding_data.external_light_begin,
                                    external_light_end: shader_binding_data.external_light_end,
                                    world_from_camera: world_from_camera.into_homogeneous_matrix().into_transform(),
                                    fov_size_at_unit_z,
                                    sample_index,
                                    max_segment_count: self.params.max_bounces + 2,
                                    render_color_space: self.params.render_color_space.into_integer(),
                                    mis_heuristic: self.params.mis_heuristic.into_integer(),
                                    flags: path_trace_flags,
                                }
                            },
                            top_level_accel,
                            sample_image_view,
                            &temp_image_views,
                        );

                        let device = &context.device;

                        let shader_binding_table_address =
                            unsafe { device.get_buffer_device_address_helper(shader_binding_table_buffer) };

                        unsafe {
                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::RAY_TRACING_KHR,
                                self.path_trace_pipeline,
                            );
                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::RAY_TRACING_KHR,
                                self.path_trace_pipeline_layout,
                                0,
                                &[path_trace_descriptor_set, texture_binding_descriptor_set],
                                &[],
                            );
                        }

                        let raygen_shader_binding_table = shader_binding_data
                            .raygen_region
                            .into_device_address_region(shader_binding_table_address);
                        let miss_shader_binding_table = shader_binding_data
                            .miss_region
                            .into_device_address_region(shader_binding_table_address);
                        let hit_shader_binding_table = shader_binding_data
                            .hit_region
                            .into_device_address_region(shader_binding_table_address);
                        let callable_shader_binding_table = vk::StridedDeviceAddressRegionKHR::default();
                        unsafe {
                            device.cmd_trace_rays_khr(
                                cmd,
                                &raygen_shader_binding_table,
                                &miss_shader_binding_table,
                                &hit_shader_binding_table,
                                &callable_shader_binding_table,
                                self.params.size.x,
                                self.params.size.y,
                                1,
                            );
                        }
                    }
                },
            );

            schedule.add_compute(
                command_name!("filter"),
                |params| {
                    params.add_image(temp_images.0, ImageUsage::COMPUTE_STORAGE_READ);
                    params.add_image(temp_images.0, ImageUsage::COMPUTE_STORAGE_READ);
                    params.add_image(temp_images.0, ImageUsage::COMPUTE_STORAGE_READ);
                    params.add_image(
                        self.result_image,
                        ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::COMPUTE_STORAGE_WRITE,
                    );
                },
                {
                    let sample_index = progress.next_sample_index;
                    move |params, cmd| {
                        let temp_image_views = [
                            params.get_image_view(temp_images.0),
                            params.get_image_view(temp_images.1),
                            params.get_image_view(temp_images.2),
                        ];
                        let result_image_view = params.get_image_view(self.result_image);

                        let descriptor_set = self.filter_descriptor_set_layout.write(
                            &descriptor_pool,
                            |buf: &mut FilterData| {
                                *buf = FilterData {
                                    image_size: self.params.size,
                                    sample_index,
                                    filter_type: self.params.filter_type.into_integer(),
                                };
                            },
                            sample_image_view,
                            &temp_image_views,
                            result_image_view,
                        );

                        dispatch_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            self.filter_pipeline_layout,
                            "trace/filter.comp.spv",
                            descriptor_set,
                            self.params.size.div_round_up(16),
                        );
                    }
                },
            );

            progress.next_sample_index += 1;
        }

        Some(self.result_image)
    }
}
