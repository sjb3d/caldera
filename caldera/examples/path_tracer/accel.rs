use crate::scene::*;
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use spark::{vk, Builder};
use std::sync::Arc;
use std::{mem, slice};

type PositionData = Vec3;
type IndexData = UVec3;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct AabbData {
    min: Vec3,
    max: Vec3,
}

// vk::AccelerationStructureInstanceKHR with Pod trait
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct AccelerationStructureInstance {
    transform: TransposedTransform3,
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

pub enum GeometryAccelData {
    Triangles {
        index_buffer: vk::Buffer,
        position_buffer: vk::Buffer,
    },
    Procedural {
        aabb_buffer: vk::Buffer,
    },
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ClusterElement {
    geometry_ref: GeometryRef,
    instance_refs: Vec<InstanceRef>, // stored in transform order (matches parent transform_refs)
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum PrimitiveType {
    Triangles,
    Procedural,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Cluster {
    transform_refs: Vec<TransformRef>,
    primitive_type: PrimitiveType,
    elements: Vec<ClusterElement>,
}

impl Cluster {
    fn unique_primitive_count(&self, scene: &Scene) -> usize {
        self.elements
            .iter()
            .map(|element| match scene.geometry(element.geometry_ref) {
                Geometry::TriangleMesh { indices, .. } => indices.len(),
                Geometry::Quad { .. } => 2,
                Geometry::Disc { .. } => 1,
                Geometry::Sphere { .. } => 1,
                Geometry::Mandelbulb { .. } => 1,
            })
            .sum()
    }

    fn instanced_primitive_count(&self, scene: &Scene) -> usize {
        self.transform_refs.len() * self.unique_primitive_count(scene)
    }
}

pub struct SceneClusters(Vec<Cluster>);

impl SceneClusters {
    fn new(scene: &Scene) -> Self {
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
                let primitive_type = match scene.geometry(geometry_ref) {
                    Geometry::TriangleMesh { .. } | Geometry::Quad { .. } => PrimitiveType::Triangles,
                    Geometry::Disc { .. } | Geometry::Sphere { .. } | Geometry::Mandelbulb { .. } => {
                        PrimitiveType::Procedural
                    }
                };
                Cluster {
                    transform_refs,
                    primitive_type,
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
            if let Some(prev) = merged.last_mut().filter(|prev| {
                prev.transform_refs == cluster.transform_refs && prev.primitive_type == cluster.primitive_type
            }) {
                prev.elements.append(&mut cluster.elements);
            } else {
                merged.push(cluster);
            }
        }

        Self(merged)
    }

    pub fn unique_accel_count(&self) -> usize {
        self.0.len()
    }

    pub fn instanced_accel_count(&self) -> usize {
        self.0.iter().map(|cluster| cluster.transform_refs.len()).sum()
    }

    pub fn geometry_iter(&self) -> impl Iterator<Item = &GeometryRef> {
        self.0
            .iter()
            .flat_map(|cluster| cluster.elements.iter().map(|element| &element.geometry_ref))
    }

    pub fn instance_iter(&self) -> impl Iterator<Item = &InstanceRef> {
        // iterate in shader binding table order
        self.0.iter().flat_map(|cluster| {
            (0..cluster.transform_refs.len()).flat_map({
                let elements = &cluster.elements;
                move |transform_index| {
                    elements
                        .iter()
                        .map(move |element| element.instance_refs.get(transform_index).unwrap())
                }
            })
        })
    }
}

struct BottomLevelAccel {
    buffer_id: BufferId,
}

struct TopLevelAccel {
    buffer_id: BufferId,
}

pub struct SceneAccel {
    scene: SharedScene,
    clusters: SceneClusters,
    geometry_accel_data: Vec<Option<GeometryAccelData>>,
    cluster_accel: Vec<BottomLevelAccel>,
    top_level_accel: TopLevelAccel,
}

impl SceneAccel {
    pub fn unique_bottom_level_accel_count(&self) -> usize {
        self.clusters.unique_accel_count()
    }

    pub fn instanced_bottom_level_accel_count(&self) -> usize {
        self.clusters.instanced_accel_count()
    }

    pub fn unique_primitive_count(&self) -> usize {
        self.clusters
            .0
            .iter()
            .map(|cluster| cluster.unique_primitive_count(&self.scene))
            .sum()
    }

    pub fn instanced_primitive_count(&self) -> usize {
        self.clusters
            .0
            .iter()
            .map(|cluster| cluster.instanced_primitive_count(&self.scene))
            .sum()
    }

    pub fn clusters(&self) -> &SceneClusters {
        &self.clusters
    }

    pub fn geometry_accel_data(&self, geometry_ref: GeometryRef) -> Option<&GeometryAccelData> {
        self.geometry_accel_data[geometry_ref.0 as usize].as_ref()
    }

    pub async fn new(resource_loader: ResourceLoader, scene: SharedScene, hit_group_count_per_instance: u32) -> Self {
        let clusters = SceneClusters::new(&scene);

        let geometry_accel_data =
            SceneAccel::create_geometry_accel_data(resource_loader.clone(), Arc::clone(&scene), &clusters).await;

        let mut cluster_accel = Vec::new();
        for cluster in clusters.0.iter() {
            cluster_accel.push(
                SceneAccel::create_bottom_level_accel(
                    cluster,
                    resource_loader.clone(),
                    Arc::clone(&scene),
                    &geometry_accel_data,
                )
                .await,
            );
        }

        let instance_buffer = SceneAccel::create_instance_buffer(
            resource_loader.clone(),
            Arc::clone(&scene),
            &clusters,
            &cluster_accel,
            hit_group_count_per_instance,
        )
        .await;

        let top_level_accel =
            SceneAccel::create_top_level_accel(resource_loader.clone(), &clusters, &cluster_accel, instance_buffer)
                .await;

        Self {
            scene,
            clusters,
            geometry_accel_data,
            cluster_accel,
            top_level_accel,
        }
    }

    async fn create_geometry_accel_data(
        resource_loader: ResourceLoader,
        scene: SharedScene,
        clusters: &SceneClusters,
    ) -> Vec<Option<GeometryAccelData>> {
        // make vertex/index buffers for each referenced geometry
        let mut tasks = Vec::new();
        for &geometry_ref in clusters.geometry_iter() {
            let loader = resource_loader.clone();
            let scene = Arc::clone(&scene);
            tasks.push(spawn(async move {
                let geometry = scene.geometry(geometry_ref);
                match geometry {
                    Geometry::TriangleMesh { .. } | Geometry::Quad { .. } => {
                        let mut mesh_builder = TriangleMeshBuilder::new();
                        let (positions, indices) = match *scene.geometry(geometry_ref) {
                            Geometry::TriangleMesh {
                                ref positions,
                                ref indices,
                                ..
                            } => (positions.as_slice(), indices.as_slice()),
                            Geometry::Quad { local_from_quad, size } => {
                                let half_size = 0.5 * size;
                                mesh_builder = mesh_builder.with_quad(
                                    local_from_quad * Vec3::new(-half_size.x, -half_size.y, 0.0),
                                    local_from_quad * Vec3::new(half_size.x, -half_size.y, 0.0),
                                    local_from_quad * Vec3::new(half_size.x, half_size.y, 0.0),
                                    local_from_quad * Vec3::new(-half_size.x, half_size.y, 0.0),
                                );
                                (mesh_builder.positions.as_slice(), mesh_builder.indices.as_slice())
                            }
                            Geometry::Disc { .. } | Geometry::Sphere { .. } | Geometry::Mandelbulb { .. } => {
                                unreachable!()
                            }
                        };

                        let index_buffer_desc = BufferDesc::new(indices.len() * mem::size_of::<IndexData>());
                        let mut writer = loader
                            .buffer_writer(
                                &index_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .await;
                        for face in indices.iter() {
                            writer.write(face);
                        }
                        let index_buffer_id = writer.finish();

                        let position_buffer_desc = BufferDesc::new(positions.len() * mem::size_of::<PositionData>());
                        let mut writer = loader
                            .buffer_writer(
                                &position_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .await;
                        for pos in positions.iter() {
                            writer.write(pos);
                        }
                        let position_buffer_id = writer.finish();

                        GeometryAccelData::Triangles {
                            position_buffer: loader.get_buffer(position_buffer_id.await),
                            index_buffer: loader.get_buffer(index_buffer_id.await),
                        }
                    }
                    Geometry::Disc {
                        local_from_disc,
                        radius,
                    } => {
                        let centre = local_from_disc.translation;
                        let normal = local_from_disc.transform_vec3(Vec3::unit_z()).normalized();
                        let radius = (radius * local_from_disc.scale).abs();
                        let half_extent = normal.map(|s| (1.0 - s * s).max(0.0).sqrt() * radius);

                        let buffer_desc = BufferDesc::new(mem::size_of::<AabbData>());
                        let mut writer = loader
                            .buffer_writer(&buffer_desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                            .await;
                        writer.write(&AabbData {
                            min: centre - half_extent,
                            max: centre + half_extent,
                        });
                        let aabb_buffer_id = writer.finish();

                        GeometryAccelData::Procedural {
                            aabb_buffer: loader.get_buffer(aabb_buffer_id.await),
                        }
                    }
                    Geometry::Sphere { centre, radius } => {
                        let centre = *centre;
                        let radius = *radius;

                        let buffer_desc = BufferDesc::new(mem::size_of::<AabbData>());
                        let mut writer = loader
                            .buffer_writer(&buffer_desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                            .await;
                        writer.write(&AabbData {
                            min: centre - Vec3::broadcast(radius),
                            max: centre + Vec3::broadcast(radius),
                        });
                        let aabb_buffer_id = writer.finish();

                        GeometryAccelData::Procedural {
                            aabb_buffer: loader.get_buffer(aabb_buffer_id.await),
                        }
                    }
                    Geometry::Mandelbulb { local_from_bulb } => {
                        let centre = local_from_bulb.translation;
                        let radius = MANDELBULB_RADIUS * local_from_bulb.scale.abs();

                        let buffer_desc = BufferDesc::new(mem::size_of::<AabbData>());
                        let mut writer = loader
                            .buffer_writer(&buffer_desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                            .await;
                        writer.write(&AabbData {
                            min: centre - Vec3::broadcast(radius),
                            max: centre + Vec3::broadcast(radius),
                        });
                        let aabb_buffer_id = writer.finish();

                        GeometryAccelData::Procedural {
                            aabb_buffer: loader.get_buffer(aabb_buffer_id.await),
                        }
                    }
                }
            }));
        }

        // make vertex/index buffers for each referenced geometry
        let mut geometry_accel_data: Vec<_> = scene.geometries.iter().map(|_| None).collect();
        for (&geometry_ref, task) in clusters.geometry_iter().zip(tasks.iter_mut()) {
            geometry_accel_data[geometry_ref.0 as usize] = Some(task.await);
        }
        geometry_accel_data
    }

    async fn create_bottom_level_accel(
        cluster: &Cluster,
        resource_loader: ResourceLoader,
        scene: SharedScene,
        geometry_accel_data: &[Option<GeometryAccelData>],
    ) -> BottomLevelAccel {
        let context = resource_loader.context();

        let mut accel_geometry = Vec::new();
        let mut max_primitive_counts = Vec::new();
        let mut build_range_info = Vec::new();
        for geometry_ref in cluster.elements.iter().map(|element| element.geometry_ref) {
            let geometry = scene.geometry(geometry_ref);
            let geometry_accel_data = geometry_accel_data[geometry_ref.0 as usize].as_ref().unwrap();

            match geometry_accel_data {
                GeometryAccelData::Triangles {
                    index_buffer,
                    position_buffer,
                } => {
                    let position_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(*position_buffer) };
                    let index_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(*index_buffer) };

                    let (vertex_count, triangle_count) = match geometry {
                        Geometry::TriangleMesh { positions, indices, .. } => {
                            (positions.len() as u32, indices.len() as u32)
                        }
                        Geometry::Quad { .. } => (4, 2),
                        Geometry::Disc { .. } | Geometry::Sphere { .. } | Geometry::Mandelbulb { .. } => unreachable!(),
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
                        flags: vk::GeometryFlagsKHR::OPAQUE,
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
                GeometryAccelData::Procedural { aabb_buffer } => {
                    let aabb_buffer_address = unsafe { context.device.get_buffer_device_address_helper(*aabb_buffer) };

                    accel_geometry.push(vk::AccelerationStructureGeometryKHR {
                        geometry_type: vk::GeometryTypeKHR::AABBS,
                        geometry: vk::AccelerationStructureGeometryDataKHR {
                            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                                data: vk::DeviceOrHostAddressConstKHR {
                                    device_address: aabb_buffer_address,
                                },
                                stride: mem::size_of::<AabbData>() as vk::DeviceSize,
                                ..Default::default()
                            },
                        },
                        flags: vk::GeometryFlagsKHR::OPAQUE,
                        ..Default::default()
                    });
                    max_primitive_counts.push(1);
                    build_range_info.push(vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: 1,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    });
                }
            }
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

        let buffer_id = resource_loader
            .graphics(move |ctx: GraphicsTaskContext| {
                let context = ctx.context;
                let schedule = ctx.schedule;

                let buffer_id = schedule.create_buffer(
                    &BufferDesc::new(sizes.acceleration_structure_size as usize),
                    BufferUsage::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_WRITE
                        | BufferUsage::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_READ
                        | BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
                );
                let scratch_buffer_id = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

                schedule.add_compute(
                    command_name!("build"),
                    |params| {
                        params.add_buffer(buffer_id, BufferUsage::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_WRITE);
                        params.add_buffer(scratch_buffer_id, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
                    },
                    {
                        let accel_geometry = accel_geometry;
                        let build_range_info = build_range_info;
                        move |params, cmd| {
                            let accel = params.get_buffer_accel(buffer_id);
                            let scratch_buffer = params.get_buffer(scratch_buffer_id);

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

                buffer_id
            })
            .await;

        BottomLevelAccel { buffer_id }
    }

    async fn create_instance_buffer(
        resource_loader: ResourceLoader,
        scene: SharedScene,
        clusters: &SceneClusters,
        cluster_accel: &[BottomLevelAccel],
        hit_group_count_per_instance: u32,
    ) -> vk::Buffer {
        let context = resource_loader.context();
        let count = clusters.instanced_accel_count();

        let desc = BufferDesc::new(count * mem::size_of::<AccelerationStructureInstance>());
        let mut writer = resource_loader
            .buffer_writer(&desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
            .await;

        let mut record_offset = 0;
        for (cluster, cluster_accel) in clusters.0.iter().zip(cluster_accel.iter()) {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR {
                acceleration_structure: Some(resource_loader.get_buffer_accel(cluster_accel.buffer_id)),
                ..Default::default()
            };
            let acceleration_structure_reference =
                unsafe { context.device.get_acceleration_structure_device_address_khr(&info) };

            for transform_ref in cluster.transform_refs.iter().copied() {
                let custom_index = transform_ref.0 & 0x00_ff_ff_ff;
                let transform = scene.transform(transform_ref);
                let instance = AccelerationStructureInstance {
                    transform: transform.world_from_local.into_transform().transposed(),
                    instance_custom_index_and_mask: 0xff_00_00_00 | custom_index,
                    instance_shader_binding_table_record_offset_and_flags: record_offset,
                    acceleration_structure_reference,
                };
                writer.write(&instance);
                record_offset += (cluster.elements.len() as u32) * hit_group_count_per_instance;
            }
        }

        let instance_buffer_id = writer.finish();
        resource_loader.get_buffer(instance_buffer_id.await)
    }

    async fn create_top_level_accel(
        resource_loader: ResourceLoader,
        clusters: &SceneClusters,
        cluster_accel: &[BottomLevelAccel],
        instance_buffer: vk::Buffer,
    ) -> TopLevelAccel {
        let context = resource_loader.context();
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

        let instance_count = clusters.instanced_accel_count() as u32;
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

        let buffer_id = resource_loader
            .graphics({
                let accel_buffer_ids: Vec<_> = cluster_accel.iter().map(|accel| accel.buffer_id).collect();
                move |ctx: GraphicsTaskContext| {
                    let context = ctx.context;
                    let schedule = ctx.schedule;

                    let buffer_id = schedule.create_buffer(
                        &BufferDesc::new(sizes.acceleration_structure_size as usize),
                        BufferUsage::TOP_LEVEL_ACCELERATION_STRUCTURE_WRITE
                            | BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
                    );
                    let scratch_buffer_id =
                        schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

                    schedule.add_compute(
                        command_name!("build"),
                        |params| {
                            for &buffer_id in accel_buffer_ids.iter() {
                                params.add_buffer(buffer_id, BufferUsage::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_READ);
                            }
                            params.add_buffer(buffer_id, BufferUsage::TOP_LEVEL_ACCELERATION_STRUCTURE_WRITE);
                            params.add_buffer(scratch_buffer_id, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
                        },
                        move |params, cmd| {
                            let accel = params.get_buffer_accel(buffer_id);
                            let scratch_buffer = params.get_buffer(scratch_buffer_id);

                            let scratch_buffer_address =
                                unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

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

                    buffer_id
                }
            })
            .await;

        TopLevelAccel { buffer_id }
    }

    pub fn top_level_buffer_id(&self) -> BufferId {
        self.top_level_accel.buffer_id
    }

    pub fn declare_parameters(&self, params: &mut RenderParameterDeclaration) {
        for bottom_level_accel in self.cluster_accel.iter() {
            params.add_buffer(
                bottom_level_accel.buffer_id,
                BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
            );
        }
        params.add_buffer(
            self.top_level_accel.buffer_id,
            BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
        );
    }
}
