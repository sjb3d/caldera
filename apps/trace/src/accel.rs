use crate::scene::*;
use bytemuck::{Pod, Zeroable};
use caldera::*;
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

pub enum GeometryBufferData {
    Triangles {
        index_buffer: StaticBufferHandle,
        position_buffer: StaticBufferHandle,
    },
    Sphere {
        aabb_buffer: StaticBufferHandle,
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
    Spheres,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Cluster {
    transform_refs: Vec<TransformRef>,
    elements: Vec<ClusterElement>,
    primitive_type: PrimitiveType,
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
                    Geometry::Sphere { .. } => PrimitiveType::Spheres,
                };
                Cluster {
                    transform_refs,
                    elements: vec![ClusterElement {
                        geometry_ref,
                        instance_refs,
                    }],
                    primitive_type,
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

    pub fn geometry_iter(&self) -> impl Iterator<Item = &GeometryRef> {
        self.0
            .iter()
            .flat_map(|cluster| cluster.elements.iter().map(|element| &element.geometry_ref))
    }

    pub fn instance_iter(&self) -> impl Iterator<Item = &InstanceRef> {
        self.0
            .iter()
            .flat_map(|cluster| cluster.elements.iter())
            .flat_map(|element| element.instance_refs.iter())
    }

    pub fn instances_grouped_by_transform_iter(&self) -> impl Iterator<Item = &InstanceRef> {
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
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

struct TopLevelAccel {
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

pub struct SceneAccel {
    context: Arc<Context>,
    scene: Arc<Scene>,
    clusters: Arc<SceneClusters>,
    geometry_buffer_data: Vec<Option<GeometryBufferData>>,
    cluster_accel: Vec<BottomLevelAccel>,
    instance_buffer: Option<StaticBufferHandle>,
    top_level_accel: Option<TopLevelAccel>,
}

impl SceneAccel {
    pub fn clusters(&self) -> &Arc<SceneClusters> {
        &self.clusters
    }

    pub fn geometry_buffer_data(&self, geometry_ref: GeometryRef) -> Option<&GeometryBufferData> {
        self.geometry_buffer_data[geometry_ref.0 as usize].as_ref()
    }

    pub fn new(context: &Arc<Context>, scene: &Arc<Scene>, resource_loader: &mut ResourceLoader) -> Self {
        let clusters = Arc::new(SceneClusters::new(scene));

        // make vertex/index buffers for each referenced geometry
        let mut geometry_buffer_data: Vec<_> = scene.geometries.iter().map(|_| None).collect();
        for geometry_ref in clusters.geometry_iter().cloned() {
            let geometry = scene.geometry(geometry_ref);
            *geometry_buffer_data.get_mut(geometry_ref.0 as usize).unwrap() = Some(match geometry {
                Geometry::TriangleMesh { .. } | Geometry::Quad { .. } => {
                    let index_buffer = resource_loader.create_buffer();
                    let position_buffer = resource_loader.create_buffer();
                    resource_loader.async_load({
                        let scene = Arc::clone(scene);
                        move |allocator| {
                            let mut mesh_builder = TriangleMeshBuilder::new();
                            let (positions, indices) = match *scene.geometry(geometry_ref) {
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
                                Geometry::Sphere { .. } => unreachable!(),
                            };

                            let index_buffer_desc = BufferDesc::new(indices.len() * mem::size_of::<IndexData>());
                            let mut writer = allocator
                                .map_buffer(
                                    index_buffer,
                                    &index_buffer_desc,
                                    BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                                        | BufferUsage::RAY_TRACING_STORAGE_READ,
                                )
                                .unwrap();
                            for face in indices.iter() {
                                writer.write(face);
                            }

                            let position_buffer_desc =
                                BufferDesc::new(positions.len() * mem::size_of::<PositionData>());
                            let mut writer = allocator
                                .map_buffer(
                                    position_buffer,
                                    &position_buffer_desc,
                                    BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                                        | BufferUsage::RAY_TRACING_STORAGE_READ,
                                )
                                .unwrap();
                            for pos in positions.iter() {
                                writer.write(pos);
                            }
                        }
                    });
                    GeometryBufferData::Triangles {
                        position_buffer,
                        index_buffer,
                    }
                }
                Geometry::Sphere { centre, radius } => {
                    let aabb_buffer = resource_loader.create_buffer();
                    resource_loader.async_load({
                        let centre = *centre;
                        let radius = *radius;
                        move |allocator| {
                            let buffer_desc = BufferDesc::new(mem::size_of::<AabbData>());
                            let mut writer = allocator
                                .map_buffer(
                                    aabb_buffer,
                                    &buffer_desc,
                                    BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT,
                                )
                                .unwrap();
                            writer.write(&AabbData {
                                min: centre - Vec3::broadcast(radius),
                                max: centre + Vec3::broadcast(radius),
                            });
                        }
                    });
                    GeometryBufferData::Sphere { aabb_buffer }
                }
            });
        }

        Self {
            context: Arc::clone(context),
            scene: Arc::clone(scene),
            clusters,
            geometry_buffer_data,
            cluster_accel: Vec::new(),
            instance_buffer: None,
            top_level_accel: None,
        }
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
            let geometry = self.scene.geometry(geometry_ref);
            let geometry_buffer_data = self.geometry_buffer_data[geometry_ref.0 as usize].as_ref().unwrap();

            match geometry_buffer_data {
                GeometryBufferData::Triangles {
                    index_buffer,
                    position_buffer,
                } => {
                    let position_buffer = resource_loader.get_buffer(*position_buffer)?;
                    let index_buffer = resource_loader.get_buffer(*index_buffer)?;

                    let position_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(position_buffer) };
                    let index_buffer_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

                    let (vertex_count, triangle_count) = match geometry {
                        Geometry::TriangleMesh { positions, indices } => (positions.len() as u32, indices.len() as u32),
                        Geometry::Quad { .. } => (4, 2),
                        Geometry::Sphere { .. } => unimplemented!(),
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
                GeometryBufferData::Sphere { aabb_buffer } => {
                    let aabb_buffer = resource_loader.get_buffer(*aabb_buffer)?;

                    let aabb_buffer_address = unsafe { context.device.get_buffer_device_address_helper(aabb_buffer) };

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
                        flags: vk::GeometryFlagsKHR::empty(),
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

    fn create_instance_buffer(
        &self,
        resource_loader: &mut ResourceLoader,
        hit_group_count_per_instance: u32,
    ) -> StaticBufferHandle {
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
            let scene = Arc::clone(&self.scene);
            let clusters = Arc::clone(&self.clusters);
            move |allocator| {
                let count = scene.instances.len();

                let desc = BufferDesc::new(count * mem::size_of::<AccelerationStructureInstance>());
                let mut writer = allocator
                    .map_buffer(instance_buffer, &desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                    .unwrap();

                let mut record_offset = 0;
                for (cluster, acceleration_structure_reference) in
                    clusters.0.iter().zip(accel_device_addresses.iter().cloned())
                {
                    for transform_ref in cluster.transform_refs.iter().cloned() {
                        let custom_index = transform_ref.0 & 0x00_ff_ff_ff;
                        let transform = scene.transform(transform_ref);
                        let instance = AccelerationStructureInstance {
                            transform: transform.0.into_transform().transposed(),
                            instance_custom_index_and_mask: 0xff_00_00_00 | custom_index,
                            instance_shader_binding_table_record_offset_and_flags: record_offset,
                            acceleration_structure_reference,
                        };
                        writer.write(&instance);
                        record_offset += (cluster.elements.len() as u32) * hit_group_count_per_instance;
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

        let instance_count = self.scene.instances.len() as u32;
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
        hit_group_count_per_instance: u32,
    ) {
        // make all bottom level acceleration structures
        while self.cluster_accel.len() < self.clusters.0.len() {
            if let Some(bottom_level_accel) = self.create_bottom_level_accel(
                self.clusters.0.get(self.cluster_accel.len()).unwrap(),
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

        // make instance buffer from transforms
        if self.instance_buffer.is_none() && self.cluster_accel.len() == self.clusters.0.len() {
            self.instance_buffer = Some(self.create_instance_buffer(resource_loader, hit_group_count_per_instance));
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

    pub fn is_ready(&self) -> bool {
        self.top_level_accel.is_some()
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
