use crate::accel::*;
use crate::scene::*;
use crate::RenderColorSpace;
use bytemuck::{Contiguous, Pod, Zeroable};
use caldera::*;
use spark::vk;
use std::{mem, slice};
use std::{ops::BitOrAssign, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct QuadLightRecord {
    emission: Vec3,
    unit_value: f32,
    area_pdf: f32,
    normal_ws: Vec3,
    corner_ws: Vec3,
    edge0_ws: Vec3,
    edge1_ws: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct SphereLightRecord {
    emission: Vec3,
    unit_value: f32,
    centre_ws: Vec3,
    radius_ws: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PathTraceData {
    world_from_camera: Transform3,
    fov_size_at_unit_z: Vec2,
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
#[derive(Clone, Copy, Zeroable, Pod, Default)]
struct ExtendShaderFlags(u32);

impl ExtendShaderFlags {
    const BSDF_TYPE_DIFFUSE: ExtendShaderFlags = ExtendShaderFlags(0x0);
    const BSDF_TYPE_MIRROR: ExtendShaderFlags = ExtendShaderFlags(0x1);
    const IS_EMISSIVE: ExtendShaderFlags = ExtendShaderFlags(0x2);
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
    light_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendTriangleHitRecord {
    index_buffer_address: u64,
    position_buffer_address: u64,
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
    shader: ExtendShader,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendMissRecord {
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
    QuadLightEval,
    QuadLightSample,
    SphereLightEval,
    SphereLightSample,
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
    callable_region: ShaderBindingRegion,
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
    const HIT_ENTRY_COUNT_PER_INSTANCE: u32 = 2;
    const MISS_ENTRY_COUNT: u32 = 2;
    const CALLABLE_ENTRY_COUNT_PER_LIGHT: u32 = 2;

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
                ShaderGroup::QuadLightEval => RayTracingShaderGroupDesc::Callable("trace/quad_light_eval.rcall.spv"),
                ShaderGroup::QuadLightSample => {
                    RayTracingShaderGroupDesc::Callable("trace/quad_light_sample.rcall.spv")
                }
                ShaderGroup::SphereLightEval => {
                    RayTracingShaderGroupDesc::Callable("trace/sphere_light_eval.rcall.spv")
                }
                ShaderGroup::SphereLightSample => {
                    RayTracingShaderGroupDesc::Callable("trace/sphere_light_sample.rcall.spv")
                }
            })
            .collect();
        let path_trace_pipeline = pipeline_cache.get_ray_tracing(&group_desc, path_trace_pipeline_layout);

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

        // gather all the lights
        // TODO: light objects too
        let emissive_instance_count = self
            .accel
            .clusters()
            .instance_iter()
            .cloned()
            .filter(|&instance_ref| {
                let shader_ref = self.scene.instance(instance_ref).shader_ref;
                self.scene.shader(shader_ref).emission.is_some()
            })
            .count();
        let total_light_count = emissive_instance_count;
        assert_eq!(total_light_count, 1);

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
        let miss_record_size = mem::size_of::<ExtendMissRecord>() as u32;
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

        // callable shaders
        let callable_record_size = mem::size_of::<QuadLightRecord>().max(mem::size_of::<SphereLightRecord>()) as u32;
        let callable_stride = align_up(
            rtpp.shader_group_handle_size + callable_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let callable_entry_count = (total_light_count as u32) * Self::CALLABLE_ENTRY_COUNT_PER_LIGHT;
        let callable_region = ShaderBindingRegion {
            offset: next_offset,
            stride: callable_stride,
            size: callable_stride * callable_entry_count,
        };
        next_offset += align_up(callable_region.size, rtpp.shader_group_base_alignment);

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
                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handles[ShaderGroup::ExtendMiss.into_integer()]);
                    writer.write_zeros(end_offset - writer.written());

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(shader_group_handles[ShaderGroup::OcclusionMiss.into_integer()]);
                    writer.write_zeros(end_offset - writer.written());
                }

                let mut light_indexer = 0u32..;
                writer.write_zeros(hit_region.offset as usize - writer.written());
                for instance_ref in clusters.instances_grouped_by_transform_iter().cloned() {
                    let instance = scene.instance(instance_ref);
                    let shader_desc = scene.shader(instance.shader_ref);

                    let mut shader = match shader_desc.surface {
                        Surface::Diffuse { reflectance } => ExtendShader {
                            flags: ExtendShaderFlags::BSDF_TYPE_DIFFUSE,
                            reflectance: reflectance / PI,
                            ..Default::default()
                        },
                        Surface::Mirror { reflectance } => ExtendShader {
                            flags: ExtendShaderFlags::BSDF_TYPE_MIRROR,
                            reflectance: Vec3::broadcast(reflectance),
                            ..Default::default()
                        },
                    };
                    if shader_desc.emission.is_some() {
                        shader.flags |= ExtendShaderFlags::IS_EMISSIVE;
                        shader.light_index = light_indexer.next().unwrap();
                    }

                    match geometry_records[instance.geometry_ref.0 as usize].unwrap() {
                        GeometryRecordData::Triangles {
                            index_buffer_address,
                            position_buffer_address,
                        } => {
                            let hit_record = ExtendTriangleHitRecord {
                                index_buffer_address,
                                position_buffer_address,
                                shader,
                                _pad: 0,
                            };

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::ExtendHitTriangle.into_integer()]);
                            writer.write(&hit_record);
                            writer.write_zeros(end_offset - writer.written());

                            let end_offset = writer.written() + hit_region.stride as usize;
                            writer.write(shader_group_handles[ShaderGroup::OcclusionHitTriangle.into_integer()]);
                            writer.write_zeros(end_offset - writer.written());
                        }
                        GeometryRecordData::Sphere => {
                            let geom = match scene.geometry(instance.geometry_ref) {
                                Geometry::Sphere { centre, radius } => SphereGeomData {
                                    centre: *centre,
                                    radius: *radius,
                                },
                                _ => unreachable!(),
                            };
                            let hit_record = ExtendSphereHitRecord { geom, shader };

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
                assert_eq!(light_indexer.next().unwrap(), total_light_count as u32);

                writer.write_zeros(callable_region.offset as usize - writer.written());
                for instance_ref in clusters.instances_grouped_by_transform_iter().cloned() {
                    let instance = scene.instance(instance_ref);
                    let shader = scene.shader(instance.shader_ref);
                    if let Some(emission) = shader.emission {
                        let world_from_local = scene.transform(instance.transform_ref).0;
                        match scene.geometry(instance.geometry_ref) {
                            Geometry::TriangleMesh { .. } => unimplemented!(),
                            Geometry::Quad {
                                size,
                                transform: quad_from_local,
                            } => {
                                let world_from_quad = world_from_local * *quad_from_local;

                                let centre_ws = world_from_quad.translation;
                                let edge0_ws = world_from_quad.transform_vec3(Vec3::new(size.x, 0.0, 0.0));
                                let edge1_ws = world_from_quad.transform_vec3(Vec3::new(0.0, size.y, 0.0));
                                let normal_ws = world_from_quad.transform_vec3(Vec3::unit_z()).normalized();

                                let unit_value =
                                    (centre_ws.abs() + 0.5 * (edge0_ws.abs() + edge1_ws.abs())).component_max();
                                let area_ws = world_from_quad.scale * world_from_quad.scale * size.x * size.y;

                                let light_record = QuadLightRecord {
                                    emission,
                                    unit_value,
                                    area_pdf: 1.0 / area_ws,
                                    normal_ws,
                                    corner_ws: centre_ws - 0.5 * (edge0_ws + edge1_ws),
                                    edge0_ws,
                                    edge1_ws,
                                };

                                let end_offset = writer.written() + callable_region.stride as usize;
                                writer.write(shader_group_handles[ShaderGroup::QuadLightEval.into_integer()]);
                                writer.write(&light_record);
                                writer.write_zeros(end_offset - writer.written());

                                let end_offset = writer.written() + callable_region.stride as usize;
                                writer.write(shader_group_handles[ShaderGroup::QuadLightSample.into_integer()]);
                                writer.write(&light_record);
                                writer.write_zeros(end_offset - writer.written());
                            }
                            Geometry::Sphere { centre, radius } => {
                                let centre_ws = world_from_local * *centre;
                                let radius_ws = world_from_local.scale.abs() * *radius;
                                let unit_value = centre_ws.abs().component_max() + radius_ws;

                                let light_record = SphereLightRecord {
                                    emission,
                                    unit_value,
                                    centre_ws,
                                    radius_ws,
                                };

                                let end_offset = writer.written() + callable_region.stride as usize;
                                writer.write(shader_group_handles[ShaderGroup::SphereLightEval.into_integer()]);
                                writer.write(&light_record);
                                writer.write_zeros(end_offset - writer.written());

                                let end_offset = writer.written() + callable_region.stride as usize;
                                writer.write(shader_group_handles[ShaderGroup::SphereLightSample.into_integer()]);
                                writer.write(&light_record);
                                writer.write_zeros(end_offset - writer.written());
                            }
                        }
                    }
                }
            }
        });
        Some(ShaderBindingTable {
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
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
            Self::HIT_ENTRY_COUNT_PER_INSTANCE,
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

        let path_trace_descriptor_set = self.path_trace_descriptor_set_layout.write(
            descriptor_pool,
            |buf: &mut PathTraceData| {
                *buf = PathTraceData {
                    world_from_camera: world_from_camera.into_homogeneous_matrix().into_transform(),
                    fov_size_at_unit_z,
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
        let callable_shader_binding_table = shader_binding_table
            .callable_region
            .into_device_address_region(shader_binding_table_address);
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
