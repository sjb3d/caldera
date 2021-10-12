use crate::accel::*;
use crate::prelude::*;
use crate::scene::*;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::prelude::*;
use imgui::{CollapsingHeader, Drag, ProgressBar, Slider, StyleColor, Ui};
use rand::{prelude::*, rngs::SmallRng};
use rayon::prelude::*;
use spark::vk;
use std::{
    collections::HashMap,
    fs::File,
    io, mem,
    ops::{BitOr, BitOrAssign},
    path::{Path, PathBuf},
    sync::Arc,
};
use structopt::StructOpt;
use strum::{EnumString, EnumVariantNames, VariantNames};

trait UnitScale {
    fn unit_scale(&self, world_from_local: Similarity3) -> f32;
}

impl UnitScale for Geometry {
    fn unit_scale(&self, world_from_local: Similarity3) -> f32 {
        let local_offset = match self {
            Geometry::TriangleMesh { min, max, .. } => max.abs().max_by_component(min.abs()).component_max(),
            Geometry::Quad { local_from_quad, size } => {
                local_from_quad.translation.abs().component_max()
                    + local_from_quad.scale.abs() * size.abs().component_max()
            }
            Geometry::Disc {
                local_from_disc,
                radius,
            } => local_from_disc.translation.abs().component_max() + local_from_disc.scale.abs() * radius.abs(),
            Geometry::Sphere { centre, radius } => centre.abs().component_max() + radius,
            Geometry::Mandelbulb { local_from_bulb } => {
                local_from_bulb.translation.abs().component_max() + local_from_bulb.scale.abs() * MANDELBULB_RADIUS
            }
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
    TriangleMesh,
    Quad,
    Disc,
    Sphere,
    Dome,
    SolidAngle,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Contiguous, PartialEq, Eq, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum SequenceType {
    Pmj,
    Sobol,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct LightInfoEntry {
    light_flags: u32,
    probability: f32,
    params_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct TriangleMeshLightParams {
    alias_table_address: u64,
    index_buffer_address: u64,
    position_buffer_address: u64,
    world_from_local: Transform3,
    illuminant_tint: Vec3,
    triangle_count: u32,
    area_pdf: f32,
    unit_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PlanarLightParams {
    illuminant_tint: Vec3,
    unit_scale: f32,
    area_pdf: f32,
    normal_ws: Vec3,
    point_ws: Vec3,
    vec0_ws: Vec3,
    vec1_ws: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SphereLightParams {
    illuminant_tint: Vec3,
    unit_scale: f32,
    centre_ws: Vec3,
    radius_ws: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct DomeLightParams {
    illuminant_tint: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SolidAngleLightParams {
    illuminant_tint: Vec3,
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
    const SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE: PathTraceFlags = PathTraceFlags(0x8);
    const PLANAR_LIGHTS_ARE_TWO_SIDED: PathTraceFlags = PathTraceFlags(0x10);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Contiguous, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum MultipleImportanceHeuristic {
    None,
    Balance,
    Power2,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CameraParams {
    world_from_local: Transform3,
    fov_size_at_unit_z: Vec2,
    aperture_radius_ls: f32,
    focus_distance_ls: f32,
    pixel_size_at_unit_z: f32,
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
    camera: CameraParams,
    sample_index: u32,
    max_segment_count: u32,
    mis_heuristic: u32,
    sequence_type: u32,
    wavelength_sampling_method: u32,
    flags: PathTraceFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct LightAliasEntry {
    split: f32,
    indices: u32,
}

descriptor_set!(PathTraceDescriptorSet {
    path_trace_uniforms: UniformData<PathTraceUniforms>,
    accel: AccelerationStructure,
    pmj_samples: StorageImage,
    sobol_samples: StorageImage,
    illuminants: CombinedImageSampler,
    conductors: CombinedImageSampler,
    smits_table: CombinedImageSampler,
    wavelength_inv_cdf: CombinedImageSampler,
    wavelength_pdf: CombinedImageSampler,
    xyz_matching: CombinedImageSampler,
    result: [StorageImage; 3],
    linear_sampler: Sampler,
});

#[repr(transparent)]
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct TextureIndex(u16);

#[derive(Clone, Copy, Default)]
struct ShaderData {
    reflectance_texture_id: Option<BindlessId>,
}

#[derive(Clone, Copy, Default)]
struct GeometryAttribRequirements {
    normal_buffer: bool,
    uv_buffer: bool,
    alias_table: bool,
}

#[derive(Clone, Copy, Default)]
struct GeometryAttribData {
    normal_buffer: Option<vk::Buffer>,
    uv_buffer: Option<vk::Buffer>,
    alias_table: Option<vk::Buffer>,
}

#[derive(Clone, Copy)]
enum GeometryRecordData {
    Triangles {
        index_buffer_address: u64,
        position_buffer_address: u64,
        normal_buffer_address: u64,
        uv_buffer_address: u64,
        alias_table_address: u64,
    },
    Procedural,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
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
    RoughDielectric,
    SmoothPlastic,
    RoughPlastic,
    RoughConductor,
}

const MIN_ROUGHNESS: f32 = 0.03;

trait IntoExtendShader {
    fn bsdf_type(&self) -> BsdfType;
    fn material_index(&self) -> u32;
    fn roughness(&self) -> f32;
}

impl IntoExtendShader for Surface {
    fn bsdf_type(&self) -> BsdfType {
        match self {
            Surface::None => BsdfType::None,
            Surface::Diffuse => BsdfType::Diffuse,
            Surface::Mirror => BsdfType::Mirror,
            Surface::SmoothDielectric => BsdfType::SmoothDielectric,
            Surface::RoughDielectric { .. } => BsdfType::RoughDielectric,
            Surface::SmoothPlastic => BsdfType::SmoothPlastic,
            Surface::RoughPlastic { .. } => BsdfType::RoughPlastic,
            Surface::RoughConductor { .. } => BsdfType::RoughConductor,
        }
    }

    fn material_index(&self) -> u32 {
        match self {
            Surface::None
            | Surface::Diffuse
            | Surface::Mirror
            | Surface::SmoothDielectric
            | Surface::SmoothPlastic
            | Surface::RoughDielectric { .. }
            | Surface::RoughPlastic { .. } => 0,
            Surface::RoughConductor { conductor, .. } => conductor.into_integer(),
        }
    }

    fn roughness(&self) -> f32 {
        match self {
            Surface::None | Surface::Diffuse => 1.0,
            Surface::Mirror | Surface::SmoothDielectric | Surface::SmoothPlastic => 0.0,
            Surface::RoughDielectric { roughness }
            | Surface::RoughPlastic { roughness }
            | Surface::RoughConductor { roughness, .. } => roughness.max(MIN_ROUGHNESS),
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Zeroable, Pod, Default)]
struct ExtendShaderFlags(u32);

impl ExtendShaderFlags {
    const HAS_NORMALS: ExtendShaderFlags = ExtendShaderFlags(0x0100_0000);
    const HAS_TEXTURE: ExtendShaderFlags = ExtendShaderFlags(0x0200_0000);
    const IS_EMISSIVE: ExtendShaderFlags = ExtendShaderFlags(0x0400_0000);
    const IS_CHECKERBOARD: ExtendShaderFlags = ExtendShaderFlags(0x0800_0000);

    fn new(bsdf_type: BsdfType, material_index: u32, texture_id: Option<BindlessId>) -> Self {
        let mut flags = Self((material_index << 20) | (bsdf_type.into_integer() << 16));
        if let Some(texture_id) = texture_id {
            flags |= Self(texture_id.index as u32) | Self::HAS_TEXTURE;
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

impl ExtendShader {
    fn new(reflectance: &Reflectance, surface: &Surface, texture_id: Option<BindlessId>) -> Self {
        let mut flags = ExtendShaderFlags::new(surface.bsdf_type(), surface.material_index(), texture_id);
        let reflectance = match reflectance {
            Reflectance::Checkerboard(c) => {
                flags |= ExtendShaderFlags::IS_CHECKERBOARD;
                *c
            }
            Reflectance::Constant(c) => *c,
            Reflectance::Texture(_) => Vec3::zero(),
        }
        .clamped(Vec3::zero(), Vec3::one());
        Self {
            flags,
            reflectance,
            roughness: surface.roughness(),
            light_index: 0,
        }
    }
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
struct ProceduralHitRecordHeader {
    unit_scale: f32,
    shader: ExtendShader,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct IntersectDiscRecord {
    header: ProceduralHitRecordHeader,
    centre: Vec3,
    normal: Vec3,
    radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct IntersectSphereRecord {
    header: ProceduralHitRecordHeader,
    centre: Vec3,
    radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct IntersectMandelbulbRecord {
    header: ProceduralHitRecordHeader,
    centre: Vec3,
    // TODO: local transform
}

#[repr(usize)]
#[derive(Clone, Copy, Contiguous)]
enum ShaderGroup {
    RayGenerator,
    ExtendMiss,
    ExtendHitTriangle,
    ExtendHitDisc,
    ExtendHitSphere,
    ExtendHitMandelbulb,
    OcclusionMiss,
    OcclusionHitTriangle,
    OcclusionHitDisc,
    OcclusionHitSphere,
    OcclusionHitMandelbulb,
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
    shader_binding_table: vk::Buffer,
    light_info_table: vk::Buffer,
    sampled_light_count: u32,
    external_light_begin: u32,
    external_light_end: u32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Contiguous, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum ToneMapMethod {
    None,
    FilmicSrgb,
    AcesFit,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Contiguous, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum FilterType {
    Box,
    Gaussian,
    Mitchell,
}

pub fn try_bool_from_str(s: &str) -> Result<bool, String> {
    match s {
        "enable" => Ok(true),
        "disable" => Ok(false),
        _ => Err(format!("{:?} is not one of enable/disable.", s)),
    }
}

type BoolParam = bool;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Contiguous, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum WavelengthSamplingMethod {
    Uniform,
    HeroMIS,
    ContinuousMIS,
}

#[derive(Debug, StructOpt)]
pub struct RendererParams {
    /// Image width
    #[structopt(short, long, default_value = "1920", global = true, display_order = 1)]
    pub width: u32,

    /// Image height
    #[structopt(short, long, default_value = "1080", global = true, display_order = 2)]
    pub height: u32,

    /// Maximum number eye path bounces
    #[structopt(short = "b", long, default_value = "8", global = true)]
    pub max_bounces: u32,

    /// Roughness accumulates along eye paths
    #[structopt(long, parse(try_from_str=try_bool_from_str), default_value="enable", global=true)]
    pub accumulate_roughness: BoolParam,

    /// Sample sphere lights by solid angle from the target point
    #[structopt(long, parse(try_from_str=try_bool_from_str), default_value="enable", global=true)]
    pub sphere_lights_sample_solid_angle: BoolParam,

    /// Quad and disc lights emit light from both sides
    #[structopt(long, parse(try_from_str=try_bool_from_str), default_value="disable", global=true)]
    pub planar_lights_are_two_sided: BoolParam,

    /// Which sampling techniques are allowed
    #[structopt(long, possible_values=SamplingTechnique::VARIANTS, default_value = "lights-and-surfaces", global=true)]
    pub sampling_technique: SamplingTechnique,

    /// How to combine samples between techniques
    #[structopt(short, long, possible_values=MultipleImportanceHeuristic::VARIANTS, default_value = "balance", global=true)]
    pub mis_heuristic: MultipleImportanceHeuristic,

    /// Image reconstruction filter
    #[structopt(short, long, possible_values=FilterType::VARIANTS, default_value = "gaussian", global=true)]
    pub filter_type: FilterType,

    /// Tone mapping method
    #[structopt(short, long, possible_values=ToneMapMethod::VARIANTS, default_value = "aces-fit", global=true)]
    pub tone_map_method: ToneMapMethod,

    /// Exposure bias
    #[structopt(
        name = "exposure-bias",
        short,
        long,
        allow_hyphen_values = true,
        default_value = "0",
        global = true
    )]
    pub log2_exposure_scale: f32,

    /// Override the camera vertical field of view
    #[structopt(name = "fov", long, global = true)]
    pub fov_y_override: Option<f32>,

    #[structopt(name = "sample-count-log2", short, long, default_value = "8", global = true)]
    pub log2_sample_count: u32,

    #[structopt(long, default_value = "sobol", global = true)]
    pub sequence_type: SequenceType,

    #[structopt(long, global = true)]
    pub d65_observer: bool,

    #[structopt(long, possible_values=WavelengthSamplingMethod::VARIANTS, default_value = "continuous-mis", global = true)]
    pub wavelength_sampling_method: WavelengthSamplingMethod,
}

impl RendererParams {
    pub fn size(&self) -> UVec2 {
        UVec2::new(self.width, self.height)
    }

    pub fn sample_count(&self) -> u32 {
        1 << self.log2_sample_count
    }

    pub fn observer_illuminant(&self) -> Illuminant {
        if self.d65_observer {
            Illuminant::D65
        } else {
            Illuminant::E
        }
    }

    pub fn observer_white_point(&self) -> WhitePoint {
        self.observer_illuminant().white_point()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct FilterData {
    image_size: UVec2,
    sequence_type: u32,
    sample_index: u32,
    filter_type: u32,
}

descriptor_set!(FilterDescriptorSet {
    data: UniformData<FilterData>,
    pmj_samples: StorageImage,
    sobol_samples: StorageImage,
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

    pub fn fraction(&self, params: &RendererParams) -> f32 {
        (self.next_sample_index as f32) / (params.sample_count() as f32)
    }

    pub fn done(&self, params: &RendererParams) -> bool {
        self.next_sample_index >= params.sample_count()
    }
}

struct AliasTable {
    under: Vec<(u32, f32)>,
    over: Vec<(u32, f32)>,
}

impl AliasTable {
    pub fn new() -> Self {
        Self {
            under: Vec::new(),
            over: Vec::new(),
        }
    }

    pub fn push(&mut self, u: f32) {
        let i = (self.under.len() + self.over.len()) as u32;
        if u < 1.0 {
            self.under.push((i, u));
        } else {
            self.over.push((i, u));
        }
    }

    pub fn into_iter(self) -> AliasTableIterator {
        AliasTableIterator { table: self }
    }
}

struct AliasTableIterator {
    table: AliasTable,
}

impl Iterator for AliasTableIterator {
    type Item = (f32, u32, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((small_i, small_u)) = self.table.under.pop() {
            if let Some((big_i, big_u)) = self.table.over.pop() {
                let remain_u = big_u - (1.0 - small_u);
                if remain_u < 1.0 {
                    self.table.under.push((big_i, remain_u));
                } else {
                    self.table.over.push((big_i, remain_u));
                }
                Some((small_u, small_i, big_i))
            } else {
                println!("no alias found for {} (entry {})", small_u, small_i);
                Some((1.0, small_i, small_i))
            }
        } else if let Some((big_i, big_u)) = self.table.over.pop() {
            if big_u > 1.0 {
                println!("no alias found for {} (entry {})", big_u, big_i);
            }
            Some((1.0, big_i, big_i))
        } else {
            None
        }
    }
}

pub struct Renderer {
    context: SharedContext,
    accel: SceneAccel,

    path_trace_pipeline_layout: vk::PipelineLayout,
    path_trace_pipeline: vk::Pipeline,

    clamp_point_sampler: vk::Sampler,
    clamp_linear_sampler: vk::Sampler,
    linear_sampler: vk::Sampler,

    shader_binding_data: ShaderBindingData,

    pmj_samples_image_view: vk::ImageView,
    sobol_samples_image_view: vk::ImageView,
    smits_table_image_view: vk::ImageView,
    illuminants_image_view: vk::ImageView,
    conductors_image_view: vk::ImageView,
    wavelength_inv_cdf_image_view: vk::ImageView,
    wavelength_pdf_image_view: vk::ImageView,
    xyz_matching_image_view: vk::ImageView,
    result_image_id: ImageId,

    pub params: RendererParams,
}

fn image_load_from_reader<R>(reader: &mut R) -> (stb::image::Info, Vec<u8>)
where
    R: io::Read + io::Seek,
{
    let (info, data) = stb::image::stbi_load_from_reader(reader, stb::image::Channels::RgbAlpha).unwrap();
    (info, data.into_vec())
}

async fn image_load(resource_loader: ResourceLoader, filename: &Path) -> BindlessId {
    let mut reader = File::open(filename).unwrap();
    let (info, data) = image_load_from_reader(&mut reader);
    let can_bc_compress = (info.width % 4) == 0 && (info.height % 4) == 0;
    let size_in_pixels = UVec2::new(info.width as u32, info.height as u32);
    println!(
        "loaded {:?}: {}x{} ({})",
        filename.file_name().unwrap(),
        size_in_pixels.x,
        size_in_pixels.y,
        if can_bc_compress { "bc1" } else { "rgba" }
    );

    let image_id = if can_bc_compress {
        // compress on write
        let image_desc = ImageDesc::new_2d(
            size_in_pixels,
            vk::Format::BC1_RGB_SRGB_BLOCK,
            vk::ImageAspectFlags::COLOR,
        );
        let mut writer = resource_loader
            .image_writer(&image_desc, ImageUsage::RAY_TRACING_SAMPLED)
            .await;
        let size_in_blocks = size_in_pixels / 4;
        let mut tmp_rgba = [0xffu8; 16 * 4];
        let mut dst_bc1 = [0u8; 8];
        for block_y in 0..size_in_blocks.y {
            for block_x in 0..size_in_blocks.x {
                for pixel_y in 0..4 {
                    let tmp_offset = (pixel_y * 4 * 4) as usize;
                    let tmp_row = &mut tmp_rgba[tmp_offset..(tmp_offset + 16)];
                    let src_offset = (((block_y * 4 + pixel_y) * size_in_pixels.x + block_x * 4) * 4) as usize;
                    let src_row = &data.as_slice()[src_offset..(src_offset + 16)];
                    for pixel_x in 0..4 {
                        for component in 0..3 {
                            let row_offset = (4 * pixel_x + component) as usize;
                            unsafe { *tmp_row.get_unchecked_mut(row_offset) = *src_row.get_unchecked(row_offset) };
                        }
                    }
                }
                stb::dxt::stb_compress_dxt_block(&mut dst_bc1, &tmp_rgba, 0, stb::dxt::CompressionMode::Normal);
                writer.write(&dst_bc1);
            }
        }
        writer.finish().await
    } else {
        // load uncompressed
        let image_desc = ImageDesc::new_2d(size_in_pixels, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR);
        let mut writer = resource_loader
            .image_writer(&image_desc, ImageUsage::RAY_TRACING_SAMPLED)
            .await;
        writer.write(data.as_slice());
        writer.finish().await
    };
    resource_loader.get_image_bindless_id(image_id)
}

impl Renderer {
    const PMJ_SEQUENCE_COUNT: u32 = 1024;

    const LOG2_MAX_SAMPLES_PER_SEQUENCE: u32 = if cfg!(debug_assertions) { 10 } else { 12 };
    const MAX_SAMPLES_PER_SEQUENCE: u32 = 1 << Self::LOG2_MAX_SAMPLES_PER_SEQUENCE;

    const HIT_ENTRY_COUNT_PER_INSTANCE: u32 = 2;
    const MISS_ENTRY_COUNT: u32 = 2;

    #[allow(clippy::too_many_arguments)]
    pub async fn new(resource_loader: ResourceLoader, scene: SharedScene, mut params: RendererParams) -> Self {
        let context = resource_loader.context();
        let accel = SceneAccel::new(
            resource_loader.clone(),
            Arc::clone(&scene),
            Self::HIT_ENTRY_COUNT_PER_INSTANCE,
        )
        .await;

        // sanitise parameters
        params.log2_sample_count = params.log2_sample_count.min(Self::LOG2_MAX_SAMPLES_PER_SEQUENCE);

        // load textures and attributes for all meshes
        let mut geometry_attrib_req = vec![GeometryAttribRequirements::default(); scene.geometries.len()];
        let mut shader_needs_texture = vec![false; scene.materials.len()];
        for instance_ref in accel.clusters().instance_iter().copied() {
            let instance = scene.instance(instance_ref);
            let material_ref = instance.material_ref;
            let geometry_ref = instance.geometry_ref;
            let material = scene.material(material_ref);
            let geometry = scene.geometry(geometry_ref);

            let has_normals = matches!(geometry, Geometry::TriangleMesh { normals: Some(_), .. });
            let has_uvs_and_texture = matches!(geometry, Geometry::TriangleMesh { uvs: Some(_), .. })
                && matches!(material.reflectance, Reflectance::Texture(_));
            let has_emissive_triangles =
                matches!(geometry, Geometry::TriangleMesh { .. }) && material.emission.is_some();

            let req = &mut geometry_attrib_req[geometry_ref.0 as usize];
            req.normal_buffer |= has_normals;
            req.uv_buffer |= has_uvs_and_texture;
            req.alias_table |= has_emissive_triangles;

            shader_needs_texture[material_ref.0 as usize] |= has_uvs_and_texture;
        }

        let mut geometry_attrib_tasks = Vec::new();
        for geometry_ref in scene.geometry_ref_iter() {
            geometry_attrib_tasks.push({
                let loader = resource_loader.clone();
                let scene = Arc::clone(&scene);
                let req = geometry_attrib_req[geometry_ref.0 as usize];
                spawn(async move {
                    let normal_buffer = if req.normal_buffer {
                        let normals = match scene.geometry(geometry_ref) {
                            Geometry::TriangleMesh {
                                normals: Some(normals), ..
                            } => normals.as_slice(),
                            _ => unreachable!(),
                        };

                        let buffer_desc = BufferDesc::new(normals.len() * mem::size_of::<Vec3>());
                        let mut writer = loader
                            .buffer_writer(&buffer_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                            .await;
                        for n in normals.iter() {
                            writer.write(n);
                        }
                        let normal_buffer_id = writer.finish();
                        Some(loader.get_buffer(normal_buffer_id.await))
                    } else {
                        None
                    };

                    let uv_buffer = if req.uv_buffer {
                        let uvs = match scene.geometry(geometry_ref) {
                            Geometry::TriangleMesh { uvs: Some(uvs), .. } => uvs.as_slice(),
                            _ => unreachable!(),
                        };

                        let buffer_desc = BufferDesc::new(uvs.len() * mem::size_of::<Vec2>());
                        let mut writer = loader
                            .buffer_writer(&buffer_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                            .await;
                        for uv in uvs.iter() {
                            writer.write(uv);
                        }
                        let uv_buffer_id = writer.finish();
                        Some(loader.get_buffer(uv_buffer_id.await))
                    } else {
                        None
                    };

                    let alias_table = if req.alias_table {
                        let (positions, indices, total_area) = match scene.geometry(geometry_ref) {
                            Geometry::TriangleMesh {
                                positions,
                                indices,
                                area,
                                ..
                            } => (positions, indices, area),
                            _ => unreachable!(),
                        };
                        let triangle_count = indices.len();

                        let mut alias_table_tmp = AliasTable::new();
                        for tri in indices.iter() {
                            let p0 = positions[tri.x as usize];
                            let p1 = positions[tri.y as usize];
                            let p2 = positions[tri.z as usize];
                            let area = 0.5 * (p2 - p1).cross(p0 - p1).mag();
                            let p = area / total_area;
                            alias_table_tmp.push(p * (triangle_count as f32));
                        }

                        let alias_table_desc = BufferDesc::new(triangle_count * mem::size_of::<LightAliasEntry>());
                        let mut writer = loader
                            .buffer_writer(&alias_table_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                            .await;
                        for (u, i, j) in alias_table_tmp.into_iter() {
                            let entry = LightAliasEntry {
                                split: u,
                                indices: (j << 16) | i,
                            };
                            writer.write(&entry);
                        }
                        let alias_table_id = writer.finish();

                        Some(loader.get_buffer(alias_table_id.await))
                    } else {
                        None
                    };

                    GeometryAttribData {
                        normal_buffer,
                        uv_buffer,
                        alias_table,
                    }
                })
            });
        }

        let mut task_index_paths = HashMap::<PathBuf, usize>::new();
        let mut shader_data_task_index = Vec::new();
        let mut texture_bindless_index_tasks = Vec::new();
        for material_ref in scene.material_ref_iter() {
            let task_index = if shader_needs_texture[material_ref.0 as usize] {
                let material = scene.material(material_ref);
                let filename = match &material.reflectance {
                    Reflectance::Texture(filename) => filename,
                    _ => unreachable!(),
                };
                Some(*task_index_paths.entry(filename.clone()).or_insert_with(|| {
                    let task_index = texture_bindless_index_tasks.len();
                    texture_bindless_index_tasks.push(spawn({
                        let loader = resource_loader.clone();
                        let filename = filename.clone();
                        async move { image_load(loader, &filename).await }
                    }));
                    task_index
                }))
            } else {
                None
            };
            shader_data_task_index.push(task_index);
        }

        let mut geometry_attrib_data = Vec::new();
        for task in geometry_attrib_tasks.iter_mut() {
            geometry_attrib_data.push(task.await);
        }
        let mut texture_bindless_index = Vec::new();
        for task in texture_bindless_index_tasks.iter_mut() {
            texture_bindless_index.push(task.await);
        }

        let shader_data: Vec<_> = shader_data_task_index
            .iter()
            .map(|task_index| ShaderData {
                reflectance_texture_id: task_index.map(|task_index| texture_bindless_index[task_index]),
            })
            .collect();

        let clamp_point_sampler = {
            let create_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::NEAREST,
                min_filter: vk::Filter::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            };
            unsafe { context.device.create_sampler(&create_info, None) }.unwrap()
        };
        let clamp_linear_sampler = {
            let create_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            };
            unsafe { context.device.create_sampler(&create_info, None) }.unwrap()
        };
        let linear_sampler = {
            let create_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                ..Default::default()
            };
            unsafe { context.device.create_sampler(&create_info, None) }.unwrap()
        };

        // make pipeline
        let pipeline_future = resource_loader.graphics({
            let bindless_descriptor_set_layout = resource_loader.bindless_descriptor_set_layout();
            move |ctx: GraphicsTaskContext| {
                let descriptor_pool = ctx.descriptor_pool;
                let pipeline_cache = ctx.pipeline_cache;
                let path_trace_descriptor_set_layout = PathTraceDescriptorSet::layout(descriptor_pool);
                let path_trace_pipeline_layout = pipeline_cache
                    .get_pipeline_layout(&[path_trace_descriptor_set_layout, bindless_descriptor_set_layout]);
                let group_desc: Vec<_> = (ShaderGroup::MIN_VALUE..=ShaderGroup::MAX_VALUE)
                    .map(|i| match ShaderGroup::from_integer(i).unwrap() {
                        ShaderGroup::RayGenerator => {
                            RayTracingShaderGroupDesc::Raygen("path_tracer/path_trace.rgen.spv")
                        }
                        ShaderGroup::ExtendMiss => RayTracingShaderGroupDesc::Miss("path_tracer/extend.rmiss.spv"),
                        ShaderGroup::ExtendHitTriangle => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/extend_triangle.rchit.spv",
                            any_hit: None,
                            intersection: None,
                        },
                        ShaderGroup::ExtendHitDisc => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/extend_procedural.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/disc.rint.spv"),
                        },
                        ShaderGroup::ExtendHitSphere => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/extend_procedural.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/sphere.rint.spv"),
                        },
                        ShaderGroup::ExtendHitMandelbulb => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/extend_procedural.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/mandelbulb.rint.spv"),
                        },
                        ShaderGroup::OcclusionMiss => {
                            RayTracingShaderGroupDesc::Miss("path_tracer/occlusion.rmiss.spv")
                        }
                        ShaderGroup::OcclusionHitTriangle => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/occlusion.rchit.spv",
                            any_hit: None,
                            intersection: None,
                        },
                        ShaderGroup::OcclusionHitDisc => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/occlusion.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/disc.rint.spv"),
                        },
                        ShaderGroup::OcclusionHitSphere => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/occlusion.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/sphere.rint.spv"),
                        },
                        ShaderGroup::OcclusionHitMandelbulb => RayTracingShaderGroupDesc::Hit {
                            closest_hit: "path_tracer/occlusion.rchit.spv",
                            any_hit: None,
                            intersection: Some("path_tracer/mandelbulb.rint.spv"),
                        },
                    })
                    .collect();
                let path_trace_pipeline = pipeline_cache.get_ray_tracing(&group_desc, path_trace_pipeline_layout);
                (path_trace_pipeline_layout, path_trace_pipeline)
            }
        });

        let loader = resource_loader.clone();
        let pmj_samples_image_view = spawn(async move {
            let sequences: Vec<Vec<_>> = (0..Self::PMJ_SEQUENCE_COUNT)
                .into_par_iter()
                .map(|i| {
                    let mut rng = SmallRng::seed_from_u64(i as u64);
                    pmj::generate(Self::MAX_SAMPLES_PER_SEQUENCE as usize, 4, &mut rng)
                })
                .collect();

            let desc = ImageDesc::new_2d(
                UVec2::new(Self::MAX_SAMPLES_PER_SEQUENCE, Self::PMJ_SEQUENCE_COUNT),
                vk::Format::R32G32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader
                .image_writer(
                    &desc,
                    ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::RAY_TRACING_STORAGE_READ,
                )
                .await;
            for sample in sequences.iter().flat_map(|sequence| sequence.iter()) {
                let pixel: [f32; 2] = [sample.x(), sample.y()];
                writer.write(&pixel);
            }
            loader.get_image_view(writer.finish().await)
        });

        let loader = resource_loader.clone();
        let sobol_samples_image_view = spawn(async move {
            let desc = ImageDesc::new_2d(
                UVec2::new(Self::MAX_SAMPLES_PER_SEQUENCE, 1),
                vk::Format::R32G32B32A32_UINT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader
                .image_writer(
                    &desc,
                    ImageUsage::COMPUTE_STORAGE_READ | ImageUsage::RAY_TRACING_STORAGE_READ,
                )
                .await;

            let seq0 = sobol(0);
            let seq1 = sobol(1);
            let seq2 = sobol(2);
            let seq3 = sobol(3);

            for (((s0, s1), s2), s3) in seq0
                .zip(seq1)
                .zip(seq2)
                .zip(seq3)
                .take(Self::MAX_SAMPLES_PER_SEQUENCE as usize)
            {
                let pixel: [u32; 4] = [s0, s1, s2, s3];
                writer.write(&pixel);
            }
            loader.get_image_view(writer.finish().await)
        });

        let loader = resource_loader.clone();
        let smits_table_image_view = spawn(async move {
            let desc = ImageDesc::new_2d(
                UVec2::new(2, 10),
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader.image_writer(&desc, ImageUsage::RAY_TRACING_SAMPLED).await;

            // c.f. "An RGB to Spectrum Conversion for Reflectances"
            #[rustfmt::skip]
            const SMITS_TABLE: &[f32] = &[
                // whi  cyan    magenta yellow  red     green   blue    pad
                1.0000, 0.9710, 1.0000, 0.0001, 0.1012, 0.0000, 1.0000, 0.0,
                1.0000, 0.9426, 1.0000, 0.0000, 0.0515, 0.0000, 1.0000, 0.0,
                0.9999, 1.0007, 0.9685, 0.1088, 0.0000, 0.0273, 0.8916, 0.0,
                0.9993, 1.0007, 0.2229, 0.6651, 0.0000, 0.7937, 0.3323, 0.0,
                0.9992, 1.0007, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0,
                0.9998, 1.0007, 0.0458, 1.0000, 0.0000, 0.9418, 0.0000, 0.0,
                1.0000, 0.1564, 0.8369, 0.9996, 0.8325, 0.1719, 0.0003, 0.0,
                1.0000, 0.0000, 1.0000, 0.9586, 1.0149, 0.0000, 0.0369, 0.0,
                1.0000, 0.0000, 1.0000, 0.9685, 1.0149, 0.0000, 0.0483, 0.0,
                1.0000, 0.0000, 0.9959, 0.9840, 1.0149, 0.0025, 0.0496, 0.0,
            ];
            writer.write(SMITS_TABLE);

            loader.get_image_view(writer.finish().await)
        });

        const WAVELENGTH_MIN: u32 = 380;
        const WAVELENGTH_MAX: u32 = 720;

        let illuminant_samples = |illuminant: Illuminant| match illuminant {
            Illuminant::E => E_ILLUMINANT,
            Illuminant::CornellBox => CORNELL_BOX_ILLUMINANT,
            Illuminant::D65 => D65_ILLUMINANT,
            Illuminant::F10 => F10_ILLUMINANT,
        };

        const ILLUMINANT_RESOLUTION: u32 = 5;
        const ILLUMINANT_PIXEL_COUNT: u32 = (WAVELENGTH_MAX - WAVELENGTH_MIN) / ILLUMINANT_RESOLUTION;
        let loader = resource_loader.clone();
        let illuminants_image_view = spawn(async move {
            let desc = ImageDesc::new_1d(
                ILLUMINANT_PIXEL_COUNT,
                vk::Format::R32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            )
            .with_layer_count(Illuminant::MAX_VALUE);

            let mut writer = loader.image_writer(&desc, ImageUsage::RAY_TRACING_SAMPLED).await;

            let wavelength_from_index = |index| {
                ((WAVELENGTH_MIN + index * ILLUMINANT_RESOLUTION) as f32) + 0.5 * (ILLUMINANT_RESOLUTION as f32)
            };

            assert_eq!(Illuminant::E.into_integer(), 0);
            for i in 1..=Illuminant::MAX_VALUE {
                let samples = illuminant_samples(Illuminant::from_integer(i).unwrap());
                let mut sweep = samples.iter().into_sweep();
                for wavelength in (0..ILLUMINANT_PIXEL_COUNT).map(wavelength_from_index) {
                    writer.write(&sweep.next(wavelength));
                }
            }

            loader.get_image_view(writer.finish().await)
        });

        const CONDUCTOR_RESOLUTION: u32 = 10;
        const CONDUCTOR_PIXEL_COUNT: u32 = (WAVELENGTH_MAX - WAVELENGTH_MIN) / CONDUCTOR_RESOLUTION;
        let loader = resource_loader.clone();
        let conductors_image_view = spawn(async move {
            let desc = ImageDesc::new_1d(
                CONDUCTOR_PIXEL_COUNT,
                vk::Format::R32G32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            )
            .with_layer_count(1 + Conductor::MAX_VALUE - Conductor::MIN_VALUE);

            let mut writer = loader.image_writer(&desc, ImageUsage::RAY_TRACING_SAMPLED).await;

            let wavelength_from_index =
                |index| ((WAVELENGTH_MIN + index * CONDUCTOR_RESOLUTION) as f32) + 0.5 * (CONDUCTOR_RESOLUTION as f32);

            for conductor in (Conductor::MIN_VALUE..=Conductor::MAX_VALUE).map(|i| Conductor::from_integer(i).unwrap())
            {
                let samples = match conductor {
                    Conductor::Aluminium => ALUMINIUM_SAMPLES,
                    Conductor::AluminiumAntimonide => ALUMINIUM_ANTIMONIDE_SAMPLES,
                    Conductor::Chromium => CHROMIUM_SAMPLES,
                    Conductor::Copper => COPPER_SAMPLES,
                    Conductor::Iron => IRON_SAMPLES,
                    Conductor::Lithium => LITHIUM_SAMPLES,
                    Conductor::Gold => GOLD_SAMPLES,
                    Conductor::Silver => SILVER_SAMPLES,
                    Conductor::TitaniumNitride => TITANIUM_NITRIDE_SAMPLES,
                    Conductor::Tungsten => TUNGSTEN_SAMPLES,
                    Conductor::Vanadium => VANADIUM_SAMPLES,
                    Conductor::VanadiumNitride => VANADIUM_NITRIDE_SAMPLES,
                    Conductor::Custom => &[
                        SampledRefractiveIndex {
                            wavelength: WAVELENGTH_MIN as f32,
                            eta: 2.0,
                            k: 0.0,
                        },
                        SampledRefractiveIndex {
                            wavelength: WAVELENGTH_MAX as f32,
                            eta: 2.0,
                            k: 0.0,
                        },
                    ],
                };
                let mut eta_sweep = samples.iter().map(|s| (s.wavelength, s.eta)).into_sweep();
                let mut k_sweep = samples.iter().map(|s| (s.wavelength, s.k)).into_sweep();
                for wavelength in (0..CONDUCTOR_PIXEL_COUNT).map(wavelength_from_index) {
                    writer.write(&eta_sweep.next(wavelength));
                    writer.write(&k_sweep.next(wavelength));
                }
            }

            loader.get_image_view(writer.finish().await)
        });

        // HACK: use the first illuminant in the scene as the pdf for wavelength sampling
        let illuminant_for_pdf = scene
            .instances
            .iter()
            .find_map(|instance| {
                scene
                    .material(instance.material_ref)
                    .emission
                    .map(|emission| emission.illuminant)
            })
            .or_else(|| {
                scene.lights.iter().map(|light| match light {
                Light::Dome { emission } => emission,
                Light::SolidAngle { emission, .. } => emission,
            }.illuminant).next()
            })
            .unwrap_or(Illuminant::E);

        const WAVELENGTH_SAMPLER_RESOLUTION: u32 = 1024;
        let loader = resource_loader.clone();
        let wavelength_pdf_image_views = spawn(async move {
            let mut sweep = illuminant_samples(illuminant_for_pdf).iter().into_sweep();
            let mut prob_samples: Vec<_> = (WAVELENGTH_MIN..=WAVELENGTH_MAX)
                .map(|wavelength| {
                    let wavelength = wavelength as f32;
                    (wavelength, sweep.next(wavelength))
                })
                .collect();

            let pdf_desc = ImageDesc::new_1d(
                WAVELENGTH_MAX - WAVELENGTH_MIN,
                vk::Format::R32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader.image_writer(&pdf_desc, ImageUsage::RAY_TRACING_SAMPLED).await;

            let prob_sum = prob_samples.iter().map(|(_, p)| p).sum::<f32>();
            let uniform_prob = 1.0 / ((WAVELENGTH_MAX - WAVELENGTH_MIN) as f32);
            for (_, p) in prob_samples.iter_mut() {
                *p = (*p / prob_sum) * 0.99 + uniform_prob * 0.01;
            }

            let mut prob_sweep = prob_samples.iter().copied().into_sweep();
            for wavelength in WAVELENGTH_MIN..WAVELENGTH_MAX {
                let pdf: f32 = prob_sweep.next((wavelength as f32) + 0.5);
                writer.write(&pdf);
            }
            let pdf_image_id = writer.finish();

            let mut cdf_acc = 0.0;
            let cdf_samples: Vec<_> = prob_samples
                .iter()
                .map(|(w, p)| {
                    let cdf = cdf_acc;
                    cdf_acc += p;
                    (cdf, *w)
                })
                .collect();

            let inv_cdf_desc = ImageDesc::new_1d(
                WAVELENGTH_SAMPLER_RESOLUTION,
                vk::Format::R32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader
                .image_writer(&inv_cdf_desc, ImageUsage::RAY_TRACING_SAMPLED)
                .await;

            let mut cdf_sweep = cdf_samples.iter().copied().into_sweep();
            for i in 0..WAVELENGTH_SAMPLER_RESOLUTION {
                let p = ((i as f32) + 0.5) / (WAVELENGTH_SAMPLER_RESOLUTION as f32);
                let wavelength: f32 = cdf_sweep.next(p);
                writer.write(&wavelength);
            }
            let inv_cdf_image_id = writer.finish();

            (
                loader.get_image_view(pdf_image_id.await),
                loader.get_image_view(inv_cdf_image_id.await),
            )
        });

        let loader = resource_loader.clone();
        let xyz_matching_image_view = spawn(async move {
            let desc = ImageDesc::new_1d(
                WAVELENGTH_MAX - WAVELENGTH_MIN,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            );
            let mut writer = loader.image_writer(&desc, ImageUsage::COMPUTE_SAMPLED).await;

            let mut sweep = xyz_matching_sweep();
            for wavelength in WAVELENGTH_MIN..WAVELENGTH_MAX {
                writer.write(&sweep.next(wavelength as f32 + 0.5).xyzw());
            }
            loader.get_image_view(writer.finish().await)
        });

        let result_image_id = resource_loader.graphics({
            let size = params.size();
            move |ctx: GraphicsTaskContext| {
                let schedule = ctx.schedule;

                let desc = ImageDesc::new_2d(size, vk::Format::R32G32B32A32_SFLOAT, vk::ImageAspectFlags::COLOR);
                let usage = ImageUsage::FRAGMENT_STORAGE_READ
                    | ImageUsage::COMPUTE_STORAGE_READ
                    | ImageUsage::COMPUTE_STORAGE_WRITE;
                schedule.create_image(&desc, usage)
            }
        });

        let (path_trace_pipeline_layout, path_trace_pipeline) = pipeline_future.await;
        let shader_binding_data = Renderer::create_shader_binding_data(
            resource_loader.clone(),
            Arc::clone(&scene),
            &accel,
            &geometry_attrib_data,
            &shader_data,
            path_trace_pipeline,
        );

        let pmj_samples_image_view = pmj_samples_image_view.await;
        let sobol_samples_image_view = sobol_samples_image_view.await;
        let smits_table_image_view = smits_table_image_view.await;
        let illuminants_image_view = illuminants_image_view.await;
        let conductors_image_view = conductors_image_view.await;
        let (wavelength_pdf_image_view, wavelength_inv_cdf_image_view) = wavelength_pdf_image_views.await;
        let xyz_matching_image_view = xyz_matching_image_view.await;
        let result_image_id = result_image_id.await;
        let shader_binding_data = shader_binding_data.await;

        Self {
            context,
            accel,
            path_trace_pipeline_layout,
            path_trace_pipeline,
            clamp_point_sampler,
            clamp_linear_sampler,
            linear_sampler,
            shader_binding_data,
            pmj_samples_image_view,
            sobol_samples_image_view,
            smits_table_image_view,
            illuminants_image_view,
            conductors_image_view,
            wavelength_inv_cdf_image_view,
            wavelength_pdf_image_view,
            xyz_matching_image_view,
            result_image_id,
            params,
        }
    }

    async fn create_shader_binding_data(
        resource_loader: ResourceLoader,
        scene: SharedScene,
        accel: &SceneAccel,
        geometry_attrib_data: &[GeometryAttribData],
        shader_data: &[ShaderData],
        path_trace_pipeline: vk::Pipeline,
    ) -> ShaderBindingData {
        // gather the data we need for records
        let context = resource_loader.context();
        let mut geometry_records = vec![None; scene.geometries.len()];
        for geometry_ref in accel.clusters().geometry_iter().copied() {
            let geometry_buffer_data = accel.geometry_accel_data(geometry_ref).unwrap();
            geometry_records[geometry_ref.0 as usize] = match geometry_buffer_data {
                GeometryAccelData::Triangles {
                    index_buffer,
                    position_buffer,
                } => {
                    let index_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(*index_buffer) };
                    let position_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(*position_buffer) };

                    let attrib_data = geometry_attrib_data[geometry_ref.0 as usize];
                    let normal_buffer_address = if let Some(buffer) = attrib_data.normal_buffer {
                        unsafe { context.device.get_buffer_device_address_helper(buffer) }
                    } else {
                        0
                    };
                    let uv_buffer_address = if let Some(buffer) = attrib_data.uv_buffer {
                        unsafe { context.device.get_buffer_device_address_helper(buffer) }
                    } else {
                        0
                    };

                    let alias_table_address = if let Some(alias_table) = attrib_data.alias_table {
                        unsafe { context.device.get_buffer_device_address_helper(alias_table) }
                    } else {
                        0
                    };

                    Some(GeometryRecordData::Triangles {
                        index_buffer_address,
                        position_buffer_address,
                        normal_buffer_address,
                        uv_buffer_address,
                        alias_table_address,
                    })
                }
                GeometryAccelData::Procedural { .. } => Some(GeometryRecordData::Procedural),
            };
        }

        // grab the shader group handles
        let rtpp = &context
            .physical_device_extra_properties
            .as_ref()
            .unwrap()
            .ray_tracing_pipeline;
        let handle_size = rtpp.shader_group_handle_size as usize;
        let shader_group_count = 1 + ShaderGroup::MAX_VALUE;
        let mut shader_group_handle_data = vec![0u8; shader_group_count * handle_size];
        unsafe {
            context.device.get_ray_tracing_shader_group_handles_khr(
                path_trace_pipeline,
                0,
                shader_group_count as u32,
                &mut shader_group_handle_data,
            )
        }
        .unwrap();

        // count the number of lights
        let emissive_instance_count = accel
            .clusters()
            .instance_iter()
            .copied()
            .filter(|&instance_ref| {
                let material_ref = scene.instance(instance_ref).material_ref;
                scene.material(material_ref).emission.is_some()
            })
            .count() as u32;
        let external_light_begin = emissive_instance_count;
        let external_light_end = emissive_instance_count + scene.lights.len() as u32;
        let sampled_light_count =
            emissive_instance_count + scene.lights.iter().filter(|light| light.can_be_sampled()).count() as u32;
        let total_light_count = external_light_end;

        // figure out the layout
        let rtpp = &context
            .physical_device_extra_properties
            .as_ref()
            .unwrap()
            .ray_tracing_pipeline;

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
        let hit_record_size = mem::size_of::<ExtendTriangleHitRecord>()
            .max(mem::size_of::<IntersectDiscRecord>())
            .max(mem::size_of::<IntersectSphereRecord>())
            .max(mem::size_of::<IntersectMandelbulbRecord>()) as u32;
        let hit_stride = align_up(
            rtpp.shader_group_handle_size + hit_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let hit_entry_count = (scene.instances.len() as u32) * Self::HIT_ENTRY_COUNT_PER_INSTANCE;
        let hit_region = ShaderBindingRegion {
            offset: next_offset,
            stride: hit_stride,
            size: hit_stride * hit_entry_count,
        };
        next_offset += align_up(hit_region.size, rtpp.shader_group_base_alignment);

        let total_size = next_offset;

        // write the table
        let (shader_binding_table_id, light_info_table_id) = {
            let shader_group_handle = |group: ShaderGroup| {
                let begin = (group.into_integer() as usize) * handle_size;
                let end = begin + handle_size;
                &shader_group_handle_data[begin..end]
            };

            let shading_binding_table_desc = BufferDesc::new(total_size as usize);
            let mut writer = resource_loader
                .buffer_writer(
                    &shading_binding_table_desc,
                    BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                )
                .await;

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
            for instance_ref in accel.clusters().instance_iter().copied() {
                let instance = scene.instance(instance_ref);
                let geometry = scene.geometry(instance.geometry_ref);
                let transform = scene.transform(instance.transform_ref);
                let material = scene.material(instance.material_ref);

                let reflectance_texture = shader_data[instance.material_ref.0 as usize].reflectance_texture_id;
                let mut shader = ExtendShader::new(&material.reflectance, &material.surface, reflectance_texture);
                if let Some(emission) = material.emission {
                    shader.flags |= ExtendShaderFlags::IS_EMISSIVE;
                    shader.light_index = light_info.len() as u32;

                    let world_from_local = transform.world_from_local;
                    let unit_scale = geometry.unit_scale(world_from_local);

                    let params_offset = light_params_data.len();
                    let (light_type, area_ws) = match geometry {
                        Geometry::TriangleMesh { indices, area, .. } => {
                            let (index_buffer_address, position_buffer_address, alias_table_address) =
                                match geometry_records[instance.geometry_ref.0 as usize].unwrap() {
                                    GeometryRecordData::Triangles {
                                        index_buffer_address,
                                        position_buffer_address,
                                        alias_table_address,
                                        ..
                                    } => (index_buffer_address, position_buffer_address, alias_table_address),
                                    GeometryRecordData::Procedural => panic!("unexpected geometry record"),
                                };
                            let triangle_count = indices.len() as u32;
                            let area_ws = area * transform.world_from_local.scale * transform.world_from_local.scale;

                            light_params_data.extend_from_slice(bytemuck::bytes_of(&TriangleMeshLightParams {
                                alias_table_address,
                                index_buffer_address,
                                position_buffer_address,
                                triangle_count,
                                world_from_local: transform.world_from_local.into_transform(),
                                illuminant_tint: emission.intensity,
                                area_pdf: 1.0 / area_ws,
                                unit_scale,
                            }));

                            (LightType::TriangleMesh, area_ws)
                        }
                        Geometry::Quad { local_from_quad, size } => {
                            let world_from_quad = world_from_local * *local_from_quad;

                            let centre_ws = world_from_quad.translation;
                            let edge0_ws = world_from_quad.transform_vec3(size.x * Vec3::unit_x());
                            let edge1_ws = world_from_quad.transform_vec3(size.y * Vec3::unit_y());
                            let normal_ws = world_from_quad.transform_vec3(Vec3::unit_z()).normalized();
                            let area_ws = (world_from_quad.scale * world_from_quad.scale * size.x * size.y).abs();

                            light_params_data.extend_from_slice(bytemuck::bytes_of(&PlanarLightParams {
                                illuminant_tint: emission.intensity,
                                unit_scale,
                                area_pdf: 1.0 / area_ws,
                                normal_ws,
                                point_ws: centre_ws - 0.5 * (edge0_ws + edge1_ws),
                                vec0_ws: edge0_ws,
                                vec1_ws: edge1_ws,
                            }));

                            (LightType::Quad, area_ws)
                        }
                        Geometry::Disc {
                            local_from_disc,
                            radius,
                        } => {
                            let world_from_disc = world_from_local * *local_from_disc;
                            let radius_ws = (world_from_disc.scale * *radius).abs();
                            let area_ws = PI * radius_ws * radius_ws;

                            let centre_ws = world_from_disc.translation;
                            let radius0_ws = world_from_disc.transform_vec3(*radius * Vec3::unit_x());
                            let radius1_ws = world_from_disc.transform_vec3(*radius * Vec3::unit_y());
                            let normal_ws = world_from_disc.transform_vec3(Vec3::unit_z()).normalized();

                            light_params_data.extend_from_slice(bytemuck::bytes_of(&PlanarLightParams {
                                illuminant_tint: emission.intensity,
                                unit_scale,
                                area_pdf: 1.0 / area_ws,
                                normal_ws,
                                point_ws: centre_ws,
                                vec0_ws: radius0_ws,
                                vec1_ws: radius1_ws,
                            }));

                            (LightType::Disc, area_ws)
                        }
                        Geometry::Sphere { centre, radius } => {
                            let centre_ws = world_from_local * *centre;
                            let radius_ws = (world_from_local.scale * *radius).abs();
                            let area_ws = 4.0 * PI * radius_ws * radius_ws;

                            light_params_data.extend_from_slice(bytemuck::bytes_of(&SphereLightParams {
                                illuminant_tint: emission.intensity,
                                unit_scale,
                                centre_ws,
                                radius_ws,
                            }));

                            (LightType::Sphere, area_ws)
                        }
                        Geometry::Mandelbulb { .. } => unimplemented!(),
                    };

                    align16_vec(&mut light_params_data);
                    light_info.push(LightInfoEntry {
                        light_flags: light_type.into_integer() | (emission.illuminant.into_integer() << 8),
                        probability: area_ws * emission.intensity.luminance() * PI, // HACK
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
                        ..
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
                    GeometryRecordData::Procedural => match scene.geometry(instance.geometry_ref) {
                        Geometry::TriangleMesh { .. } | Geometry::Quad { .. } => unreachable!(),
                        Geometry::Disc {
                            local_from_disc,
                            radius,
                        } => {
                            let hit_record = IntersectDiscRecord {
                                header: ProceduralHitRecordHeader { unit_scale, shader },
                                centre: local_from_disc.translation,
                                normal: local_from_disc.transform_vec3(Vec3::unit_z()).normalized(),
                                radius: (local_from_disc.scale * radius).abs(),
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::ExtendHitDisc));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::OcclusionHitDisc));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());
                        }
                        Geometry::Sphere { centre, radius } => {
                            let hit_record = IntersectSphereRecord {
                                header: ProceduralHitRecordHeader { unit_scale, shader },
                                centre: *centre,
                                radius: radius.abs(),
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
                        Geometry::Mandelbulb { local_from_bulb } => {
                            let hit_record = IntersectMandelbulbRecord {
                                header: ProceduralHitRecordHeader { unit_scale, shader },
                                centre: local_from_bulb.translation,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::ExtendHitMandelbulb));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handle(ShaderGroup::OcclusionHitMandelbulb));
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());
                        }
                    },
                }
            }
            let shader_binding_table_id = writer.finish();

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
                let (light_type, illuminant, sampling_power) = match light {
                    Light::Dome { emission } => {
                        light_params_data.extend_from_slice(bytemuck::bytes_of(&DomeLightParams {
                            illuminant_tint: emission.intensity,
                        }));

                        (LightType::Dome, emission.illuminant, 0.0)
                    }
                    Light::SolidAngle {
                        emission,
                        direction_ws,
                        solid_angle,
                    } => {
                        let direction_ws = direction_ws.normalized();
                        let solid_angle = solid_angle.abs();

                        light_params_data.extend_from_slice(bytemuck::bytes_of(&SolidAngleLightParams {
                            illuminant_tint: emission.intensity,
                            direction_ws,
                            solid_angle: solid_angle.abs(),
                        }));

                        // HACK: use the total local light power to split samples 50/50
                        // TODO: use scene bounds to estimate the light power?
                        let fake_power = local_light_total_power.unwrap_or(1.0);
                        (LightType::SolidAngle, emission.illuminant, fake_power)
                    }
                };

                align16_vec(&mut light_params_data);
                light_info.push(LightInfoEntry {
                    light_flags: light_type.into_integer() | (illuminant.into_integer() << 8),
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
            let mut writer = resource_loader
                .buffer_writer(&light_info_table_desc, BufferUsage::RAY_TRACING_STORAGE_READ)
                .await;

            let rcp_total_sampled_light_power = 1.0 / light_info.iter().map(|info| info.probability).sum::<f32>();
            for info in light_info.iter_mut() {
                info.probability *= rcp_total_sampled_light_power;
                writer.write(info);
            }

            writer.write_zeros(light_alias_offset - writer.written());
            let mut alias_table = AliasTable::new();
            for u in light_info
                .iter()
                .map(|info| info.probability)
                .take(sampled_light_count as usize)
            {
                alias_table.push(u * (sampled_light_count as f32));
            }
            for (u, i, j) in alias_table.into_iter() {
                let entry = LightAliasEntry {
                    split: u,
                    indices: (j << 16) | i,
                };
                writer.write(&entry);
            }

            writer.write_zeros(light_params_offset - writer.written());
            writer.write(light_params_data.as_slice());

            let light_info_table_id = writer.finish();

            (shader_binding_table_id, light_info_table_id)
        };

        ShaderBindingData {
            raygen_region,
            miss_region,
            hit_region,
            shader_binding_table: resource_loader.get_buffer(shader_binding_table_id.await),
            light_info_table: resource_loader.get_buffer(light_info_table_id.await),
            sampled_light_count,
            external_light_begin,
            external_light_end,
        }
    }

    pub fn debug_ui(&mut self, progress: &mut RenderProgress, ui: &Ui) {
        let mut needs_reset = false;

        if CollapsingHeader::new("Stats").default_open(true).build(ui) {
            ui.text(format!("Unique BLAS: {}", self.accel.unique_bottom_level_accel_count()));
            ui.text(format!(
                "Instanced BLAS: {}",
                self.accel.instanced_bottom_level_accel_count()
            ));
            ui.text(format!("Unique Prims: {}", self.accel.unique_primitive_count()));
            ui.text(format!("Instanced Prims: {}", self.accel.instanced_primitive_count()));
            ui.text(format!(
                "Sampled Lights: {}",
                self.shader_binding_data.sampled_light_count
            ));
            ui.text(format!("Total Lights: {}", self.shader_binding_data.external_light_end));
        }

        if CollapsingHeader::new("Renderer").default_open(true).build(ui) {
            ProgressBar::new(progress.fraction(&self.params)).build(ui);
            Slider::new("Log2 Samples", 0, Self::LOG2_MAX_SAMPLES_PER_SEQUENCE)
                .build(ui, &mut self.params.log2_sample_count);

            ui.text("Sequence Type:");
            needs_reset |= ui.radio_button("PMJ", &mut self.params.sequence_type, SequenceType::Pmj);
            needs_reset |= ui.radio_button("Sobol", &mut self.params.sequence_type, SequenceType::Sobol);

            ui.text("Wavelength Sampling Method:");
            needs_reset |= ui.radio_button(
                "Uniform",
                &mut self.params.wavelength_sampling_method,
                WavelengthSamplingMethod::Uniform,
            );
            needs_reset |= ui.radio_button(
                "Hero MIS",
                &mut self.params.wavelength_sampling_method,
                WavelengthSamplingMethod::HeroMIS,
            );
            needs_reset |= ui.radio_button(
                "Continuous MIS",
                &mut self.params.wavelength_sampling_method,
                WavelengthSamplingMethod::ContinuousMIS,
            );

            ui.text("Sampling Technique:");
            needs_reset |= ui.radio_button(
                "Lights Only",
                &mut self.params.sampling_technique,
                SamplingTechnique::LightsOnly,
            );
            needs_reset |= ui.radio_button(
                "Surfaces Only",
                &mut self.params.sampling_technique,
                SamplingTechnique::SurfacesOnly,
            );
            needs_reset |= ui.radio_button(
                "Lights And Surfaces",
                &mut self.params.sampling_technique,
                SamplingTechnique::LightsAndSurfaces,
            );

            let id = ui.push_id("MIS Heuristic");
            if self.params.sampling_technique == SamplingTechnique::LightsAndSurfaces {
                ui.text("MIS Heuristic:");
                needs_reset |= ui.radio_button(
                    "None",
                    &mut self.params.mis_heuristic,
                    MultipleImportanceHeuristic::None,
                );
                needs_reset |= ui.radio_button(
                    "Balance",
                    &mut self.params.mis_heuristic,
                    MultipleImportanceHeuristic::Balance,
                );
                needs_reset |= ui.radio_button(
                    "Power2",
                    &mut self.params.mis_heuristic,
                    MultipleImportanceHeuristic::Power2,
                );
            } else {
                ui.text_disabled("MIS Heuristic:");
                let style = ui.push_style_color(StyleColor::Text, ui.style_color(StyleColor::TextDisabled));
                ui.radio_button_bool("None", true);
                ui.radio_button_bool("Balance", false);
                ui.radio_button_bool("Power2", false);
                style.pop();
            }
            id.pop();

            needs_reset |= Slider::new("Max Bounces", 0, 24).build(ui, &mut self.params.max_bounces);
            needs_reset |= ui.checkbox("Accumulate Roughness", &mut self.params.accumulate_roughness);
            needs_reset |= ui.checkbox(
                "Sample Sphere Light Solid Angle",
                &mut self.params.sphere_lights_sample_solid_angle,
            );
            needs_reset |= ui.checkbox(
                "Planar Lights Are Two Sided",
                &mut self.params.planar_lights_are_two_sided,
            );
        }

        if CollapsingHeader::new("Film").default_open(true).build(ui) {
            ui.text("Filter:");
            needs_reset |= ui.radio_button("Box", &mut self.params.filter_type, FilterType::Box);
            needs_reset |= ui.radio_button("Gaussian", &mut self.params.filter_type, FilterType::Gaussian);
            needs_reset |= ui.radio_button("Mitchell", &mut self.params.filter_type, FilterType::Mitchell);

            let id = ui.push_id("Tone Map");
            Drag::new("Exposure Bias")
                .speed(0.05)
                .build(ui, &mut self.params.log2_exposure_scale);
            ui.text("Tone Map:");
            ui.radio_button("None", &mut self.params.tone_map_method, ToneMapMethod::None);
            ui.radio_button(
                "Filmic sRGB",
                &mut self.params.tone_map_method,
                ToneMapMethod::FilmicSrgb,
            );
            ui.radio_button(
                "ACES (fitted)",
                &mut self.params.tone_map_method,
                ToneMapMethod::AcesFit,
            );
            id.pop();
        }

        if progress.next_sample_index > self.params.sample_count() {
            needs_reset = true;
        }
        if needs_reset {
            progress.reset();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render<'a>(
        &'a self,
        progress: &mut RenderProgress,
        context: &'a Context,
        schedule: &mut RenderSchedule<'a>,
        pipeline_cache: &'a PipelineCache,
        descriptor_pool: &'a DescriptorPool,
        camera: &Camera,
    ) -> ImageId {
        // do a pass
        if !progress.done(&self.params) {
            let temp_desc = ImageDesc::new_2d(self.params.size(), vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR);
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
                    let camera = *camera;
                    move |params, cmd| {
                        let temp_image_views = [
                            params.get_image_view(temp_images.0),
                            params.get_image_view(temp_images.1),
                            params.get_image_view(temp_images.2),
                        ];

                        let size_flt = self.params.size().as_float();
                        let aspect_ratio = size_flt.x / size_flt.y;

                        let (world_from_camera, fov_y, aperture_radius, focus_distance) = match camera {
                            Camera::Pinhole {
                                world_from_camera,
                                fov_y,
                            } => (world_from_camera, fov_y, 0.0, 2.0),
                            Camera::ThinLens {
                                world_from_camera,
                                fov_y,
                                aperture_radius,
                                focus_distance,
                            } => (world_from_camera, fov_y, aperture_radius, focus_distance),
                        };
                        let fov_size_at_unit_z = 2.0 * (0.5 * fov_y).tan() * Vec2::new(aspect_ratio, 1.0);
                        let pixel_size_at_unit_z = fov_size_at_unit_z.y / size_flt.y;

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
                        if self.params.sphere_lights_sample_solid_angle {
                            path_trace_flags |= PathTraceFlags::SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE;
                        }
                        if self.params.planar_lights_are_two_sided {
                            path_trace_flags |= PathTraceFlags::PLANAR_LIGHTS_ARE_TWO_SIDED;
                        }

                        let light_info_table_address = unsafe {
                            self.context
                                .device
                                .get_buffer_device_address_helper(self.shader_binding_data.light_info_table)
                        };
                        let align16 = |n: u64| (n + 15) & !15;
                        let light_alias_table_address = align16(
                            light_info_table_address
                                + (self.shader_binding_data.external_light_end as u64)
                                    * (mem::size_of::<LightInfoEntry>() as u64),
                        );
                        let light_params_base_address = align16(
                            light_alias_table_address
                                + (self.shader_binding_data.sampled_light_count as u64)
                                    * (mem::size_of::<LightAliasEntry>() as u64),
                        );

                        let bindless_descriptor_set = params.get_bindless_descriptor_set();
                        let path_trace_descriptor_set = PathTraceDescriptorSet::create(
                            descriptor_pool,
                            |buf: &mut PathTraceUniforms| {
                                *buf = PathTraceUniforms {
                                    light_info_table_address,
                                    light_alias_table_address,
                                    light_params_base_address,
                                    sampled_light_count: self.shader_binding_data.sampled_light_count,
                                    external_light_begin: self.shader_binding_data.external_light_begin,
                                    external_light_end: self.shader_binding_data.external_light_end,
                                    camera: CameraParams {
                                        world_from_local: world_from_camera.into_homogeneous_matrix().into_transform(),
                                        fov_size_at_unit_z,
                                        aperture_radius_ls: aperture_radius,
                                        focus_distance_ls: focus_distance,
                                        pixel_size_at_unit_z,
                                    },
                                    sample_index,
                                    max_segment_count: self.params.max_bounces + 2,
                                    mis_heuristic: self.params.mis_heuristic.into_integer(),
                                    sequence_type: self.params.sequence_type.into_integer(),
                                    wavelength_sampling_method: self.params.wavelength_sampling_method.into_integer(),
                                    flags: path_trace_flags,
                                }
                            },
                            params.get_buffer_accel(self.accel.top_level_buffer_id()),
                            self.pmj_samples_image_view,
                            self.sobol_samples_image_view,
                            self.illuminants_image_view,
                            self.clamp_linear_sampler,
                            self.conductors_image_view,
                            self.clamp_linear_sampler,
                            self.smits_table_image_view,
                            self.clamp_point_sampler,
                            self.wavelength_inv_cdf_image_view,
                            self.clamp_linear_sampler,
                            self.wavelength_pdf_image_view,
                            self.clamp_point_sampler,
                            self.xyz_matching_image_view,
                            self.clamp_linear_sampler,
                            &temp_image_views,
                            self.linear_sampler,
                        );

                        let device = &context.device;

                        let shader_binding_table_address = unsafe {
                            device.get_buffer_device_address_helper(self.shader_binding_data.shader_binding_table)
                        };

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
                                &[path_trace_descriptor_set.set, bindless_descriptor_set],
                                &[],
                            );
                        }

                        let raygen_shader_binding_table = self
                            .shader_binding_data
                            .raygen_region
                            .into_device_address_region(shader_binding_table_address);
                        let miss_shader_binding_table = self
                            .shader_binding_data
                            .miss_region
                            .into_device_address_region(shader_binding_table_address);
                        let hit_shader_binding_table = self
                            .shader_binding_data
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
                                self.params.width,
                                self.params.height,
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
                    params.add_image(temp_images.1, ImageUsage::COMPUTE_STORAGE_READ);
                    params.add_image(temp_images.2, ImageUsage::COMPUTE_STORAGE_READ);
                    params.add_image(
                        self.result_image_id,
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
                        let result_image_view = params.get_image_view(self.result_image_id);

                        let descriptor_set = FilterDescriptorSet::create(
                            descriptor_pool,
                            |buf: &mut FilterData| {
                                *buf = FilterData {
                                    image_size: self.params.size(),
                                    sequence_type: self.params.sequence_type.into_integer(),
                                    sample_index,
                                    filter_type: self.params.filter_type.into_integer(),
                                };
                            },
                            self.pmj_samples_image_view,
                            self.sobol_samples_image_view,
                            &temp_image_views,
                            result_image_view,
                        );

                        dispatch_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            "path_tracer/filter.comp.spv",
                            &[],
                            descriptor_set,
                            self.params.size().div_round_up(8),
                        );
                    }
                },
            );

            progress.next_sample_index += 1;
        }

        self.result_image_id
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_sampler(Some(self.clamp_point_sampler), None);
            self.context
                .device
                .destroy_sampler(Some(self.clamp_linear_sampler), None);
            self.context.device.destroy_sampler(Some(self.linear_sampler), None);
        }
    }
}
