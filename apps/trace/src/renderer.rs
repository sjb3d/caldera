use crate::accel::*;
use crate::scene::*;
use crate::RenderColorSpace;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::*;
use spark::vk;
use std::{mem, ops::Deref, slice};
use std::{ops::BitOrAssign, sync::Arc};

#[derive(Clone, Copy)]
struct QuadLight {
    transform: Similarity3,
    size: Vec2,
    emission: Vec3,
}

impl QuadLight {
    fn area_ws(&self) -> f32 {
        let scale = self.transform.scale;
        scale * scale * self.size.x * self.size.y
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PathTraceData {
    world_from_camera: Transform3,
    fov_size_at_unit_z: Vec2,
    world_from_light: Transform3,
    light_size: Vec2,
    light_area_ws: f32,
    light_emission: Vec3,
    sample_index: u32,
    max_segment_count: u32,
    render_color_space: u32,
    use_max_roughness: u32,
}

descriptor_set_layout!(PathTraceDescriptorSetLayout {
    data: UniformData<PathTraceData>,
    accel: AccelerationStructure,
    samples: StorageImage,
    result_r: StorageImage,
    result_g: StorageImage,
    result_b: StorageImage,
});

#[derive(Clone, Copy)]
enum GeometryRecordData {
    Triangles {
        index_buffer_address: u64,
        position_buffer_address: u64,
    },
    Sphere,
}

#[repr(transparent)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendRecordFlags(u32);

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendTriangleHitRecord {
    index_buffer_address: u64,
    position_buffer_address: u64,
    reflectance: Vec3,
    flags: ExtendRecordFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SphereHitRecord {
    centre: Vec3,
    radius: f32,
    reflectance: Vec3,
    flags: ExtendRecordFlags,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendMissRecord {
    flags: ExtendRecordFlags,
}

impl ExtendRecordFlags {
    const BSDF_TYPE_DIFFUSE: ExtendRecordFlags = ExtendRecordFlags(0x0);
    const BSDF_TYPE_MIRROR: ExtendRecordFlags = ExtendRecordFlags(0x1);
    const IS_EMISSIVE: ExtendRecordFlags = ExtendRecordFlags(0x2);

    fn empty() -> Self {
        Self(0)
    }
}

impl BitOrAssign for ExtendRecordFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
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

struct ShaderBindingTable {
    raygen_region: ShaderBindingRegion,
    miss_region: ShaderBindingRegion,
    hit_region: ShaderBindingRegion,
    buffer: StaticBufferHandle,
}

pub struct Renderer {
    context: Arc<Context>,
    scene: Arc<Scene>,
    accel: SceneAccel,
    path_trace_pipeline_layout: vk::PipelineLayout,
    path_trace_descriptor_set_layout: PathTraceDescriptorSetLayout,
    path_trace_pipeline: vk::Pipeline,
    shader_binding_table: Option<ShaderBindingTable>,
}

impl Renderer {
    const HIT_GROUP_COUNT_PER_INSTANCE: u32 = 2;
    const MISS_GROUP_COUNT: u32 = 2;

    pub fn new(
        context: &Arc<Context>,
        scene: &Arc<Scene>,
        descriptor_set_layout_cache: &mut DescriptorSetLayoutCache,
        pipeline_cache: &PipelineCache,
        resource_loader: &mut ResourceLoader,
    ) -> Self {
        let accel = SceneAccel::new(context, scene, resource_loader);

        let path_trace_descriptor_set_layout = PathTraceDescriptorSetLayout::new(descriptor_set_layout_cache);
        let path_trace_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(path_trace_descriptor_set_layout.0);

        // make pipeline
        let path_trace_pipeline = pipeline_cache.get_ray_tracing(
            &[
                RayTracingShaderGroupDesc::Raygen("trace/path_trace.rgen.spv"),
                RayTracingShaderGroupDesc::Miss("trace/extend.rmiss.spv"),
                RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/extend_triangle.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
                RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/extend_sphere.rchit.spv",
                    any_hit: None,
                    intersection: Some("trace/sphere.rint.spv"),
                },
                RayTracingShaderGroupDesc::Miss("trace/occlusion.rmiss.spv"),
                RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/occlusion.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
                RayTracingShaderGroupDesc::Hit {
                    closest_hit: "trace/occlusion.rchit.spv",
                    any_hit: None,
                    intersection: Some("trace/sphere.rint.spv"),
                },
            ],
            path_trace_pipeline_layout,
        );

        Self {
            context: Arc::clone(context),
            scene: Arc::clone(scene),
            accel,
            path_trace_descriptor_set_layout,
            path_trace_pipeline_layout,
            path_trace_pipeline,
            shader_binding_table: None,
        }
    }

    fn create_shader_binding_table(&self, resource_loader: &mut ResourceLoader) -> Option<ShaderBindingTable> {
        // gather the data we need for records
        let mut geometry_records = vec![None; self.scene.geometries.len()];
        for geometry_ref in self.accel.clusters().geometry_iter().cloned() {
            let geometry_buffer_data = self.accel.geometry_buffer_data(geometry_ref)?;
            geometry_records[geometry_ref.0 as usize] = match geometry_buffer_data {
                GeometryBufferData::Triangles {
                    index_buffer,
                    position_buffer,
                } => {
                    let index_buffer = resource_loader.get_buffer(*index_buffer)?;
                    let position_buffer = resource_loader.get_buffer(*position_buffer)?;
                    let index_buffer_address =
                        unsafe { self.context.device.get_buffer_device_address_helper(index_buffer) };
                    let position_buffer_address =
                        unsafe { self.context.device.get_buffer_device_address_helper(position_buffer) };
                    Some(GeometryRecordData::Triangles {
                        index_buffer_address,
                        position_buffer_address,
                    })
                }
                GeometryBufferData::Sphere { .. } => Some(GeometryRecordData::Sphere),
            };
        }

        // figure out the layout
        let rtpp = self.context.ray_tracing_pipeline_properties.as_ref().unwrap();

        let align_up = |n: u32, a: u32| (n + a - 1) & !(a - 1);
        let mut next_offset = 0;

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

        let miss_record_size = mem::size_of::<ExtendMissRecord>() as u32;
        let miss_stride = align_up(
            rtpp.shader_group_handle_size + miss_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let miss_entry_count = Self::MISS_GROUP_COUNT;
        let miss_region = ShaderBindingRegion {
            offset: next_offset,
            stride: miss_stride,
            size: miss_stride * miss_entry_count,
        };
        next_offset += align_up(miss_region.size, rtpp.shader_group_base_alignment);

        let hit_record_size = mem::size_of::<ExtendTriangleHitRecord>().max(mem::size_of::<SphereHitRecord>()) as u32;
        let hit_stride = align_up(
            rtpp.shader_group_handle_size + hit_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let hit_entry_count = (self.scene.instances.len() as u32) * Self::HIT_GROUP_COUNT_PER_INSTANCE;
        let hit_region = ShaderBindingRegion {
            offset: next_offset,
            stride: hit_stride,
            size: hit_stride * hit_entry_count,
        };
        next_offset += align_up(hit_region.size, rtpp.shader_group_base_alignment);

        let total_size = next_offset;

        // write the table
        let shader_binding_table = resource_loader.create_buffer();
        resource_loader.async_load({
            let path_trace_pipeline = self.path_trace_pipeline;
            let context = Arc::clone(&self.context);
            let scene = Arc::clone(&self.scene);
            let clusters = Arc::clone(self.accel.clusters());
            move |allocator| {
                let rtpp = context.ray_tracing_pipeline_properties.as_ref().unwrap();
                let handle_size = rtpp.shader_group_handle_size as usize;
                let shader_group_count = 1 + ShaderGroup::MAX_VALUE;
                let mut handle_data = vec![0u8; shader_group_count * handle_size];
                unsafe {
                    context.device.get_ray_tracing_shader_group_handles_khr(
                        path_trace_pipeline,
                        0,
                        shader_group_count as u32,
                        &mut handle_data,
                    )
                }
                .unwrap();
                let shader_group_handles: Vec<_> = handle_data.as_slice().chunks(handle_size).collect();

                let desc = BufferDesc::new(total_size as usize);
                let mut writer = allocator
                    .map_buffer(
                        shader_binding_table,
                        &desc,
                        BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                    )
                    .unwrap();

                assert_eq!(raygen_region.offset, 0);
                writer.write(shader_group_handles[ShaderGroup::RayGenerator.into_integer()]);

                writer.write_zeros(miss_region.offset as usize - writer.written());
                {
                    let mut flags = ExtendRecordFlags::empty();
                    if !scene.lights.is_empty() {
                        flags |= ExtendRecordFlags::IS_EMISSIVE;
                    }

                    let extend_miss_record = ExtendMissRecord { flags };

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handles[ShaderGroup::ExtendMiss.into_integer()]);
                    writer.write(&extend_miss_record);
                    writer.write_zeros(end_offset - writer.written());

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handles[ShaderGroup::OcclusionMiss.into_integer()]);
                    writer.write_zeros(end_offset - writer.written());
                }

                writer.write_zeros(hit_region.offset as usize - writer.written());
                for instance_ref in clusters.instances_grouped_by_transform_iter().cloned() {
                    let instance = scene.instance(instance_ref);
                    let shader = scene.shader(instance.shader_ref);

                    let (reflectance, mut flags) = match shader.surface {
                        Surface::Diffuse { reflectance } => (reflectance / PI, ExtendRecordFlags::BSDF_TYPE_DIFFUSE),
                        Surface::Mirror { reflectance } => {
                            (Vec3::broadcast(reflectance), ExtendRecordFlags::BSDF_TYPE_MIRROR)
                        }
                    };
                    if shader.emission.is_some() {
                        flags |= ExtendRecordFlags::IS_EMISSIVE;
                    }

                    match geometry_records[instance.geometry_ref.0 as usize].unwrap() {
                        GeometryRecordData::Triangles {
                            index_buffer_address,
                            position_buffer_address,
                        } => {
                            let extend_hit_record = ExtendTriangleHitRecord {
                                index_buffer_address,
                                position_buffer_address,
                                reflectance,
                                flags,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::ExtendHitTriangle.into_integer()]);
                            writer.write(&extend_hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::OcclusionHitTriangle.into_integer()]);
                            writer.write_zeros(end_offset - writer.written());
                        }
                        GeometryRecordData::Sphere => {
                            let (centre, radius) = match scene.geometry(instance.geometry_ref) {
                                Geometry::Sphere { centre, radius } => (centre, radius),
                                _ => unreachable!(),
                            };
                            let hit_record = SphereHitRecord {
                                centre: *centre,
                                radius: *radius,
                                reflectance,
                                flags,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::ExtendHitSphere.into_integer()]);
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::OcclusionHitSphere.into_integer()]);
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());
                        }
                    }
                }
            }
        });
        Some(ShaderBindingTable {
            raygen_region,
            miss_region,
            hit_region,
            buffer: shader_binding_table,
        })
    }

    pub fn update<'a>(
        &mut self,
        context: &'a Context,
        resource_loader: &mut ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) {
        // continue with acceleration structures
        self.accel.update(
            context,
            resource_loader,
            global_allocator,
            schedule,
            Self::HIT_GROUP_COUNT_PER_INSTANCE,
        );

        // make shader binding table
        if self.shader_binding_table.is_none() {
            self.shader_binding_table = self.create_shader_binding_table(resource_loader);
        }
    }

    pub fn is_ready(&self, resource_loader: &ResourceLoader) -> bool {
        self.accel.is_ready()
            && self
                .shader_binding_table
                .as_ref()
                .map(|table| resource_loader.get_buffer(table.buffer).is_some())
                .unwrap_or(false)
    }

    pub fn declare_parameters(&self, params: &mut RenderParameterDeclaration) {
        self.accel.declare_parameters(params);
    }

    pub fn emit_trace(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_pool: &DescriptorPool,
        resource_loader: &ResourceLoader,
        render_color_space: RenderColorSpace,
        max_bounces: u32,
        use_max_roughness: bool,
        sample_image_view: vk::ImageView,
        sample_index: u32,
        camera_ref: CameraRef,
        world_from_camera: Isometry3,
        result_image_views: &(vk::ImageView, vk::ImageView, vk::ImageView),
        trace_size: UVec2,
    ) {
        let camera = self.scene.camera(camera_ref);
        let aspect_ratio = (trace_size.x as f32) / (trace_size.y as f32);
        let fov_size_at_unit_z = 2.0 * (0.5 * camera.fov_y).tan() * Vec2::new(aspect_ratio, 1.0);

        let scene = self.scene.deref();
        let quad_light = scene
            .instances
            .iter()
            .filter_map(|instance| {
                let emission = scene.shader(instance.shader_ref).emission?;
                match scene.geometry(instance.geometry_ref) {
                    Geometry::TriangleMesh { .. } => None,
                    Geometry::Quad { size, transform } => Some(QuadLight {
                        transform: scene.transform(instance.transform_ref).0 * *transform,
                        size: *size,
                        emission,
                    }),
                    Geometry::Sphere { .. } => None,
                }
            })
            .next();
        let dome_light = scene.lights.first();

        let path_trace_descriptor_set = self.path_trace_descriptor_set_layout.write(
            descriptor_pool,
            |buf: &mut PathTraceData| {
                *buf = PathTraceData {
                    world_from_camera: world_from_camera.into_homogeneous_matrix().into_transform(),
                    fov_size_at_unit_z,
                    world_from_light: quad_light
                        .map(|light| light.transform.into_transform())
                        .unwrap_or_else(Transform3::identity),
                    light_size: quad_light.map(|light| light.size).unwrap_or_else(Vec2::zero),
                    light_area_ws: quad_light.map(|light| light.area_ws()).unwrap_or(0.0),
                    light_emission: quad_light
                        .map(|light| light.emission)
                        .or_else(|| dome_light.map(|light| light.emission))
                        .unwrap_or_else(Vec3::zero),
                    sample_index,
                    max_segment_count: max_bounces + 2,
                    render_color_space: render_color_space.into_integer(),
                    use_max_roughness: if use_max_roughness { 1 } else { 0 },
                }
            },
            self.accel.top_level_accel().unwrap(),
            sample_image_view,
            result_image_views.0,
            result_image_views.1,
            result_image_views.2,
        );

        let device = &self.context.device;

        let shader_binding_table = self.shader_binding_table.as_ref().unwrap();
        let shader_binding_table_buffer = resource_loader.get_buffer(shader_binding_table.buffer).unwrap();
        let shader_binding_table_address =
            unsafe { device.get_buffer_device_address_helper(shader_binding_table_buffer) };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, self.path_trace_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.path_trace_pipeline_layout,
                0,
                slice::from_ref(&path_trace_descriptor_set),
                &[],
            );
        }

        let raygen_shader_binding_table = shader_binding_table
            .raygen_region
            .into_device_address_region(shader_binding_table_address);
        let miss_shader_binding_table = shader_binding_table
            .miss_region
            .into_device_address_region(shader_binding_table_address);
        let hit_shader_binding_table = shader_binding_table
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
                trace_size.x,
                trace_size.y,
                1,
            );
        }
    }
}
