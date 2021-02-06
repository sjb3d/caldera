use crate::scene::*;
use bytemuck::{Pod, Zeroable};
use caldera::*;
use spark::{vk, Builder};
use std::{mem, slice};
use std::{ops::BitOrAssign, sync::Arc};

type PositionData = Vec3;
type IndexData = UVec3;

#[repr(transparent)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendRecordFlags(u32);

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ExtendHitRecord {
    index_buffer_address: u64,
    position_buffer_address: u64,
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

// vk::AccelerationStructureInstanceKHR with Pod trait
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct AccelerationStructureInstance {
    pub transform: TransposedTransform3,
    pub instance_custom_index_and_mask: u32,
    pub instance_shader_binding_table_record_offset_and_flags: u32,
    pub acceleration_structure_reference: u64,
}

struct TriangleMeshData {
    index_buffer: StaticBufferHandle,
    position_buffer: StaticBufferHandle,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ClusterElement {
    geometry_ref: GeometryRef,
    instance_refs: Vec<InstanceRef>, // stored in transform order (matches parent transform_refs)
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Cluster {
    transform_refs: Vec<TransformRef>,
    elements: Vec<ClusterElement>,
}

impl Cluster {
    fn instance_refs_grouped_by_transform_iter(&self) -> impl Iterator<Item = &InstanceRef> {
        (0..self.transform_refs.len()).flat_map({
            let elements = &self.elements;
            move |transform_index| {
                elements
                    .iter()
                    .map(move |element| element.instance_refs.get(transform_index).unwrap())
            }
        })
    }
}

struct BottomLevelAccel {
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

struct TopLevelAccel {
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

struct SceneShared {
    scene: Scene,
    clusters: Vec<Cluster>,
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

pub struct SceneAccel {
    shared: Arc<SceneShared>,
    context: Arc<Context>,
    path_trace_pipeline: vk::Pipeline,
    geometry_data: Vec<Option<TriangleMeshData>>,
    shader_binding_table: Option<ShaderBindingTable>,
    cluster_accel: Vec<BottomLevelAccel>,
    instance_buffer: Option<StaticBufferHandle>,
    top_level_accel: Option<TopLevelAccel>,
}

impl SceneAccel {
    const HIT_GROUP_COUNT_PER_INSTANCE: u32 = 2;
    const MISS_GROUP_COUNT: u32 = 2;

    fn make_clusters(scene: &Scene) -> Vec<Cluster> {
        // gather a vector of instances per geometry
        let mut instance_refs_per_geometry = vec![Vec::new(); scene.geometries.len()];
        for instance_ref in scene.instance_ref_iter() {
            let instance = scene.instance(instance_ref);
            instance_refs_per_geometry
                .get_mut(instance.geometry_ref.0 as usize)
                .unwrap()
                .push(instance_ref);
        }

        // convert to single-geometry clusters, with instances sorted by transform reference
        let mut clusters: Vec<_> = scene
            .geometry_ref_iter()
            .zip(instance_refs_per_geometry.drain(..))
            .filter(|(_, instance_refs)| !instance_refs.is_empty())
            .map(|(geometry_ref, instance_refs)| {
                let mut pairs: Vec<_> = instance_refs
                    .iter()
                    .map(|&instance_ref| (scene.instance(instance_ref).transform_ref, instance_ref))
                    .collect();
                pairs.sort_unstable();
                let (transform_refs, instance_refs): (Vec<_>, Vec<_>) = pairs.drain(..).unzip();
                Cluster {
                    transform_refs,
                    elements: vec![ClusterElement {
                        geometry_ref,
                        instance_refs,
                    }],
                }
            })
            .collect();

        // sort the clusters by transform set so that we can merge in a single pass
        clusters.sort_unstable();

        // merge clusters that are used with the same set of transforms
        let mut merged = Vec::<Cluster>::new();
        for mut cluster in clusters.drain(..) {
            if let Some(prev) = merged
                .last_mut()
                .filter(|prev| prev.transform_refs == cluster.transform_refs)
            {
                prev.elements.append(&mut cluster.elements);
            } else {
                merged.push(cluster);
            }
        }

        merged
    }

    pub fn new(
        scene: Scene,
        context: &Arc<Context>,
        path_trace_pipeline_layout: vk::PipelineLayout,
        pipeline_cache: &PipelineCache,
        resource_loader: &mut ResourceLoader,
    ) -> Self {
        let clusters = Self::make_clusters(&scene);
        let shared = Arc::new(SceneShared { scene, clusters });

        // make pipeline
        let path_trace_pipeline = pipeline_cache.get_ray_tracing(
            &[
                RayTracingShaderGroupDesc::Raygen("trace/path_trace.rgen.spv"),
                RayTracingShaderGroupDesc::Miss("trace/extend.rmiss.spv"),
                RayTracingShaderGroupDesc::TrianglesHit {
                    closest_hit: "trace/extend.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
                RayTracingShaderGroupDesc::Miss("trace/occlusion.rmiss.spv"),
                RayTracingShaderGroupDesc::TrianglesHit {
                    closest_hit: "trace/occlusion.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
            ],
            path_trace_pipeline_layout,
        );

        // make vertex/index buffers for each referenced geometry
        let mut geometry_data: Vec<_> = shared.scene.geometries.iter().map(|_| None).collect();
        for geometry_ref in shared
            .clusters
            .iter()
            .flat_map(|cluster| cluster.elements.iter().map(|element| element.geometry_ref))
        {
            let index_buffer = resource_loader.create_buffer();
            let position_buffer = resource_loader.create_buffer();
            resource_loader.async_load({
                let shared = Arc::clone(&shared);
                move |allocator| {
                    let mut mesh_builder = TriangleMeshBuilder::new();
                    let (positions, indices) = match *shared.scene.geometry(geometry_ref) {
                        Geometry::TriangleMesh {
                            ref positions,
                            ref indices,
                        } => (positions.as_slice(), indices.as_slice()),
                        Geometry::Quad { transform, size } => {
                            let half_size = 0.5 * size;
                            mesh_builder = mesh_builder.with_quad(
                                transform * Vec3::new(-half_size.x, -half_size.y, 0.0),
                                transform * Vec3::new(half_size.x, -half_size.y, 0.0),
                                transform * Vec3::new(half_size.x, half_size.y, 0.0),
                                transform * Vec3::new(-half_size.x, half_size.y, 0.0),
                            );
                            (mesh_builder.positions.as_slice(), mesh_builder.indices.as_slice())
                        }
                    };

                    let index_buffer_desc = BufferDesc::new(indices.len() * mem::size_of::<IndexData>());
                    let mut writer = allocator
                        .map_buffer(
                            index_buffer,
                            &index_buffer_desc,
                            BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                        )
                        .unwrap();
                    for face in indices.iter() {
                        writer.write(face);
                    }

                    let position_buffer_desc = BufferDesc::new(positions.len() * mem::size_of::<PositionData>());
                    let mut writer = allocator
                        .map_buffer(
                            position_buffer,
                            &position_buffer_desc,
                            BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                        )
                        .unwrap();
                    for pos in positions.iter() {
                        writer.write(pos);
                    }
                }
            });
            *geometry_data.get_mut(geometry_ref.0 as usize).unwrap() = Some(TriangleMeshData {
                position_buffer,
                index_buffer,
            });
        }

        Self {
            shared,
            context: Arc::clone(&context),
            path_trace_pipeline,
            geometry_data,
            shader_binding_table: None,
            cluster_accel: Vec::new(),
            instance_buffer: None,
            top_level_accel: None,
        }
    }

    fn create_shader_binding_table(&self, resource_loader: &mut ResourceLoader) -> Option<ShaderBindingTable> {
        // gather the data we need for records
        let mut geometry_record = vec![(0u64, 0u64); self.shared.scene.geometries.len()];
        for geometry_ref in self
            .shared
            .clusters
            .iter()
            .flat_map(|cluster| cluster.elements.iter().map(|element| element.geometry_ref))
        {
            let geometry_data = self.geometry_data[geometry_ref.0 as usize].as_ref().unwrap();
            let index_buffer = resource_loader.get_buffer(geometry_data.index_buffer)?;
            let position_buffer = resource_loader.get_buffer(geometry_data.position_buffer)?;
            geometry_record[geometry_ref.0 as usize] = (
                unsafe { self.context.device.get_buffer_device_address_helper(index_buffer) },
                unsafe { self.context.device.get_buffer_device_address_helper(position_buffer) },
            );
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

        let hit_record_size = mem::size_of::<ExtendHitRecord>() as u32;
        let hit_stride = align_up(
            rtpp.shader_group_handle_size + hit_record_size,
            rtpp.shader_group_handle_alignment,
        );
        let hit_entry_count = (self.shared.scene.instances.len() as u32) * Self::HIT_GROUP_COUNT_PER_INSTANCE;
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
            let shared = Arc::clone(&self.shared);
            move |allocator| {
                let rtpp = context.ray_tracing_pipeline_properties.as_ref().unwrap();
                let shader_group_count = 1 + Self::MISS_GROUP_COUNT + Self::HIT_GROUP_COUNT_PER_INSTANCE;
                let handle_size = rtpp.shader_group_handle_size as usize;
                let mut handle_data = vec![0u8; (shader_group_count as usize) * handle_size];
                unsafe {
                    context.device.get_ray_tracing_shader_group_handles_khr(
                        path_trace_pipeline,
                        0,
                        shader_group_count,
                        &mut handle_data,
                    )
                }
                .unwrap();

                let remain = handle_data.as_slice();
                let (raygen_group_handle, remain) = remain.split_at(handle_size);
                let (extend_miss_group_handle, remain) = remain.split_at(handle_size);
                let (extend_hit_group_handle, remain) = remain.split_at(handle_size);
                let (occlusion_miss_group_handle, remain) = remain.split_at(handle_size);
                let (occlusion_hit_group_handle, _remain) = remain.split_at(handle_size);

                let desc = BufferDesc::new(total_size as usize);
                let mut writer = allocator
                    .map_buffer(
                        shader_binding_table,
                        &desc,
                        BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                    )
                    .unwrap();

                assert_eq!(raygen_region.offset, 0);
                writer.write(raygen_group_handle);

                writer.write_zeros(miss_region.offset as usize - writer.written());
                {
                    let mut flags = ExtendRecordFlags::empty();
                    if !shared.scene.lights.is_empty() {
                        flags |= ExtendRecordFlags::IS_EMISSIVE;
                    }

                    let extend_miss_record = ExtendMissRecord { flags };

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(extend_miss_group_handle);
                    writer.write(&extend_miss_record);
                    writer.write_zeros(end_offset - writer.written());

                    let end_offset = writer.written() + miss_region.stride as usize;
                    writer.write(occlusion_miss_group_handle);
                    writer.write_zeros(end_offset - writer.written());
                }

                writer.write_zeros(hit_region.offset as usize - writer.written());
                for cluster in shared.clusters.iter() {
                    for instance_ref in cluster.instance_refs_grouped_by_transform_iter().cloned() {
                        let instance = shared.scene.instance(instance_ref);
                        let (index_buffer_address, position_buffer_address) =
                            geometry_record[instance.geometry_ref.0 as usize];
                        let shader = shared.scene.shader(instance.shader_ref);

                        let (reflectance, mut flags) = match shader.surface {
                            Surface::Diffuse { reflectance } => {
                                (reflectance / PI, ExtendRecordFlags::BSDF_TYPE_DIFFUSE)
                            }
                            Surface::Mirror { reflectance } => {
                                (Vec3::broadcast(reflectance), ExtendRecordFlags::BSDF_TYPE_MIRROR)
                            }
                        };
                        if shader.emission.is_some() {
                            flags |= ExtendRecordFlags::IS_EMISSIVE;
                        }

                        let extend_hit_record = ExtendHitRecord {
                            index_buffer_address,
                            position_buffer_address,
                            reflectance: reflectance,
                            flags,
                        };

                        let end_offset = writer.written() + hit_region.stride as usize;
                        writer.write(extend_hit_group_handle);
                        writer.write(&extend_hit_record);
                        writer.write_zeros(end_offset - writer.written());

                        let end_offset = writer.written() + hit_region.stride as usize;
                        writer.write(occlusion_hit_group_handle);
                        writer.write_zeros(end_offset - writer.written());
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

    fn create_bottom_level_accel<'a>(
        &self,
        cluster: &Cluster,
        context: &'a Context,
        resource_loader: &ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> Option<BottomLevelAccel> {
        let mut accel_geometry = Vec::new();
        let mut max_primitive_counts = Vec::new();
        let mut build_range_info = Vec::new();
        for geometry_ref in cluster.elements.iter().map(|element| element.geometry_ref) {
            let geometry = self.shared.scene.geometry(geometry_ref);
            let geometry_data = self.geometry_data[geometry_ref.0 as usize].as_ref().unwrap();

            let position_buffer = resource_loader.get_buffer(geometry_data.position_buffer)?;
            let index_buffer = resource_loader.get_buffer(geometry_data.index_buffer)?;

            let position_buffer_address = unsafe { context.device.get_buffer_device_address_helper(position_buffer) };
            let index_buffer_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

            let (vertex_count, triangle_count) = match geometry {
                Geometry::TriangleMesh { positions, indices } => (positions.len() as u32, indices.len() as u32),
                Geometry::Quad { .. } => (4, 2),
            };

            accel_geometry.push(vk::AccelerationStructureGeometryKHR {
                geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                geometry: vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        vertex_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: position_buffer_address,
                        },
                        vertex_stride: mem::size_of::<PositionData>() as vk::DeviceSize,
                        max_vertex: vertex_count,
                        index_type: vk::IndexType::UINT32,
                        index_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: index_buffer_address,
                        },
                        ..Default::default()
                    },
                },
                flags: vk::GeometryFlagsKHR::empty(),
                ..Default::default()
            });
            max_primitive_counts.push(triangle_count);
            build_range_info.push(vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: triangle_count,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            });
        }

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .p_geometries(&accel_geometry);

        let sizes = {
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                context.device.get_acceleration_structure_build_sizes_khr(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    Some(&max_primitive_counts),
                    &mut sizes,
                )
            };
            sizes
        };

        println!(
            "geometry count: {} (to be instanced {} times)",
            cluster.elements.len(),
            cluster.transform_refs.len()
        );
        println!("build scratch size: {}", sizes.build_scratch_size);
        println!("acceleration structure size: {}", sizes.acceleration_structure_size);

        let buffer_desc = BufferDesc::new(sizes.acceleration_structure_size as usize);
        let buffer = schedule.create_buffer(
            &buffer_desc,
            BufferUsage::ACCELERATION_STRUCTURE_WRITE
                | BufferUsage::ACCELERATION_STRUCTURE_READ
                | BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
            global_allocator,
        );
        let accel = {
            let create_info = vk::AccelerationStructureCreateInfoKHR {
                buffer: Some(schedule.get_buffer_hack(buffer)),
                size: buffer_desc.size as vk::DeviceSize,
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                ..Default::default()
            };
            unsafe { context.device.create_acceleration_structure_khr(&create_info, None) }.unwrap()
        };

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

        schedule.add_compute(
            command_name!("build"),
            |params| {
                params.add_buffer(buffer, BufferUsage::ACCELERATION_STRUCTURE_WRITE);
                params.add_buffer(scratch_buffer, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
            },
            {
                let accel_geometry = accel_geometry;
                let build_range_info = build_range_info;
                move |params, cmd| {
                    let scratch_buffer = params.get_buffer(scratch_buffer);

                    let scratch_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(Some(accel))
                        .p_geometries(&accel_geometry)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_buffer_address,
                        });

                    unsafe {
                        context.device.cmd_build_acceleration_structures_khr(
                            cmd,
                            slice::from_ref(&build_info),
                            slice::from_ref(&build_range_info.as_ptr()),
                        )
                    };
                }
            },
        );

        Some(BottomLevelAccel { accel, buffer })
    }

    pub fn create_instance_buffer(&self, resource_loader: &mut ResourceLoader) -> StaticBufferHandle {
        let accel_device_addresses: Vec<_> = self
            .cluster_accel
            .iter()
            .map(|bottom_level_accel| {
                let info = vk::AccelerationStructureDeviceAddressInfoKHR {
                    acceleration_structure: Some(bottom_level_accel.accel),
                    ..Default::default()
                };
                unsafe { self.context.device.get_acceleration_structure_device_address_khr(&info) }
            })
            .collect();

        let instance_buffer = resource_loader.create_buffer();
        resource_loader.async_load({
            let shared = Arc::clone(&self.shared);
            move |allocator| {
                let count = shared.scene.instances.len();

                let desc = BufferDesc::new(count * mem::size_of::<AccelerationStructureInstance>());
                let mut writer = allocator
                    .map_buffer(instance_buffer, &desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                    .unwrap();

                let mut record_offset = 0;
                for (cluster, acceleration_structure_reference) in
                    shared.clusters.iter().zip(accel_device_addresses.iter().cloned())
                {
                    for transform_ref in cluster.transform_refs.iter().cloned() {
                        let custom_index = transform_ref.0 & 0x00_ff_ff_ff;
                        let transform = shared.scene.transform(transform_ref);
                        let instance = AccelerationStructureInstance {
                            transform: transform.0.transposed(),
                            instance_custom_index_and_mask: 0xff_00_00_00 | custom_index,
                            instance_shader_binding_table_record_offset_and_flags: record_offset,
                            acceleration_structure_reference,
                        };
                        writer.write(&instance);
                        record_offset += (cluster.elements.len() as u32) * Self::HIT_GROUP_COUNT_PER_INSTANCE;
                    }
                }
            }
        });
        instance_buffer
    }

    fn create_top_level_accel<'a>(
        &self,
        context: &'a Context,
        instance_buffer: vk::Buffer,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> TopLevelAccel {
        let instance_buffer_address = unsafe { context.device.get_buffer_device_address_helper(instance_buffer) };

        let accel_geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_address,
                    },
                    ..Default::default()
                },
            },
            ..Default::default()
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .p_geometries(slice::from_ref(&accel_geometry));

        let instance_count = self.shared.scene.instances.len() as u32;
        let sizes = {
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                context.device.get_acceleration_structure_build_sizes_khr(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    Some(slice::from_ref(&instance_count)),
                    &mut sizes,
                )
            };
            sizes
        };

        println!("instance count: {}", instance_count);
        println!("build scratch size: {}", sizes.build_scratch_size);
        println!("acceleration structure size: {}", sizes.acceleration_structure_size);

        let buffer_desc = BufferDesc::new(sizes.acceleration_structure_size as usize);
        let buffer = schedule.create_buffer(
            &buffer_desc,
            BufferUsage::ACCELERATION_STRUCTURE_WRITE | BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
            global_allocator,
        );
        let accel = {
            let create_info = vk::AccelerationStructureCreateInfoKHR {
                buffer: Some(schedule.get_buffer_hack(buffer)),
                size: buffer_desc.size as vk::DeviceSize,
                ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                ..Default::default()
            };
            unsafe { context.device.create_acceleration_structure_khr(&create_info, None) }.unwrap()
        };

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

        schedule.add_compute(
            command_name!("build"),
            |params| {
                for bottom_level_accel in self.cluster_accel.iter() {
                    params.add_buffer(bottom_level_accel.buffer, BufferUsage::ACCELERATION_STRUCTURE_READ);
                }
                params.add_buffer(buffer, BufferUsage::ACCELERATION_STRUCTURE_WRITE);
                params.add_buffer(scratch_buffer, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
            },
            move |params, cmd| {
                let scratch_buffer = params.get_buffer(scratch_buffer);

                let scratch_buffer_address = unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

                let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(Some(accel))
                    .p_geometries(slice::from_ref(&accel_geometry))
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer_address,
                    });

                let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: instance_count,
                    primitive_offset: 0,
                    first_vertex: 0,
                    transform_offset: 0,
                };

                unsafe {
                    context.device.cmd_build_acceleration_structures_khr(
                        cmd,
                        slice::from_ref(&build_info),
                        &[&build_range_info],
                    )
                };
            },
        );

        TopLevelAccel { accel, buffer }
    }

    pub fn update<'a>(
        &mut self,
        context: &'a Context,
        resource_loader: &mut ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) {
        // make all bottom level acceleration structures
        while self.cluster_accel.len() < self.shared.clusters.len() {
            if let Some(bottom_level_accel) = self.create_bottom_level_accel(
                self.shared.clusters.get(self.cluster_accel.len()).unwrap(),
                context,
                resource_loader,
                global_allocator,
                schedule,
            ) {
                self.cluster_accel.push(bottom_level_accel);
            } else {
                break;
            }
        }

        // make shader binding table
        if self.shader_binding_table.is_none() {
            self.shader_binding_table = self.create_shader_binding_table(resource_loader);
        }

        // make instance buffer from transforms
        if self.instance_buffer.is_none() && self.cluster_accel.len() == self.shared.clusters.len() {
            self.instance_buffer = Some(self.create_instance_buffer(resource_loader));
        }

        // make the top level acceleration structure
        if self.top_level_accel.is_none() {
            if let Some(instance_buffer) = self
                .instance_buffer
                .and_then(|handle| resource_loader.get_buffer(handle))
            {
                self.top_level_accel =
                    Some(self.create_top_level_accel(context, instance_buffer, global_allocator, schedule));
            }
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.shared.scene
    }

    pub fn is_ready(&self, resource_loader: &ResourceLoader) -> bool {
        self.top_level_accel.is_some()
            && self
                .shader_binding_table
                .as_ref()
                .map(|table| resource_loader.get_buffer(table.buffer).is_some())
                .unwrap_or(false)
    }

    pub fn top_level_accel(&self) -> Option<vk::AccelerationStructureKHR> {
        self.top_level_accel.as_ref().map(|top_level| top_level.accel)
    }

    pub fn declare_parameters(&self, params: &mut RenderParameterDeclaration) {
        for bottom_level_accel in self.cluster_accel.iter() {
            params.add_buffer(
                bottom_level_accel.buffer,
                BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
            );
        }
        if let Some(top_level) = self.top_level_accel.as_ref() {
            params.add_buffer(top_level.buffer, BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE);
        }
    }

    pub fn emit_trace(
        &self,
        cmd: vk::CommandBuffer,
        resource_loader: &ResourceLoader,
        path_trace_pipeline_layout: vk::PipelineLayout,
        path_trace_descriptor_set: vk::DescriptorSet,
        dims: UVec2,
    ) {
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
                path_trace_pipeline_layout,
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
                dims.x,
                dims.y,
                1,
            );
        }
    }
}

impl Drop for SceneAccel {
    fn drop(&mut self) {
        let device = &self.context.device;
        for bottom_level_accel in self.cluster_accel.iter() {
            unsafe { device.destroy_acceleration_structure_khr(Some(bottom_level_accel.accel), None) };
        }
        if let Some(top_level_accel) = self.top_level_accel.as_ref() {
            unsafe { device.destroy_acceleration_structure_khr(Some(top_level_accel.accel), None) };
        }
    }
}
