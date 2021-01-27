use crate::scene::*;
use caldera::*;
use spark::{vk, Builder};
use std::sync::Arc;
use std::{mem, slice};

#[repr(C)]
struct PositionData([f32; 3]);

#[repr(C)]
struct IndexData([u32; 3]);

#[derive(Clone, Copy)]
struct TriangleMeshData {
    position_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
}

#[derive(Debug)]
struct Cluster {
    transform_refs: Vec<TransformRef>,
    geometry_refs: Vec<GeometryRef>,
    instance_refs: Vec<InstanceRef>,
}

struct BottomLevelAccel {
    pub accel: vk::AccelerationStructureKHR,
    pub buffer: BufferHandle,
}

pub struct SceneAccel {
    scene: Arc<Scene>,
    geometry_data: Vec<TriangleMeshData>,
    clusters: Vec<Cluster>,
    bottom_level_accel: Vec<BottomLevelAccel>,
}

impl SceneAccel {
    fn make_clusters(scene: &Scene) -> Vec<Cluster> {
        // start with one cluster per geometry
        let mut clusters: Vec<_> = scene
            .geometry_ref_iter()
            .map(|g| Cluster {
                transform_refs: Vec::new(),
                geometry_refs: vec![g],
                instance_refs: Vec::new(),
            })
            .collect();
        for instance_ref in scene.instance_ref_iter() {
            let instance = scene.instance(instance_ref).unwrap();
            let cluster = clusters.get_mut(instance.geometry_ref.0 as usize).unwrap();
            cluster.transform_refs.push(instance.transform_ref);
            cluster.instance_refs.push(instance_ref);
        }

        // sort the transform sets for each cluster so they can be compared
        for cluster in clusters.iter_mut() {
            cluster.transform_refs.sort_unstable();
        }

        // sort the clusters by transform set so that we can merge in a single pass
        clusters.sort_unstable_by(|a, b| {
            (a.transform_refs.as_slice(), a.geometry_refs.first())
                .cmp(&(b.transform_refs.as_slice(), b.geometry_refs.first()))
        });

        // merge clusters that are used with the same set of transforms
        let mut merged = Vec::<Cluster>::new();
        for cluster in clusters.drain(..) {
            if let Some(prev) = merged
                .last_mut()
                .filter(|prev| prev.transform_refs == cluster.transform_refs)
            {
                prev.geometry_refs.extend_from_slice(&cluster.geometry_refs);
                prev.instance_refs.extend_from_slice(&cluster.instance_refs);
            } else {
                merged.push(cluster);
            }
        }
        merged
    }

    pub fn new(scene: Scene, context: &Arc<Context>, resource_loader: &mut ResourceLoader) -> Self {
        let scene = Arc::new(scene);

        // TODO: make pipeline

        // make vertex/index buffers for each geometry
        let geometry_data = scene
            .geometry_ref_iter()
            .map(|geometry_ref| {
                let data = TriangleMeshData {
                    position_buffer: resource_loader.create_buffer(),
                    index_buffer: resource_loader.create_buffer(),
                };
                resource_loader.async_load({
                    let scene = Arc::clone(&scene);
                    move |allocator| {
                        let geometry = scene.geometry(geometry_ref).unwrap();
                        let position_buffer_desc =
                            BufferDesc::new(geometry.positions.len() * mem::size_of::<PositionData>());
                        let mut writer = allocator
                            .map_buffer(
                                data.position_buffer,
                                &position_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .unwrap();
                        for pos in geometry.positions.iter() {
                            writer.write_all(pos.as_byte_slice());
                        }

                        let index_buffer_desc = BufferDesc::new(geometry.indices.len() * mem::size_of::<IndexData>());
                        let mut writer = allocator
                            .map_buffer(
                                data.index_buffer,
                                &index_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .unwrap();
                        for face in geometry.indices.iter() {
                            writer.write_all(face.as_byte_slice());
                        }
                    }
                });
                data
            })
            .collect();

        let clusters = Self::make_clusters(&scene);

        Self {
            scene,
            geometry_data,
            clusters,
            bottom_level_accel: Vec::new(),
        }
    }

    fn create_bottom_level_accel<'a>(
        &self,
        cluster: &Cluster,
        context: &'a Arc<Context>,
        resource_loader: &ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> Option<BottomLevelAccel> {
        let mut accel_geometry = Vec::new();
        let mut max_primitive_counts = Vec::new();
        let mut build_range_info = Vec::new();
        for geometry_ref in cluster.geometry_refs.iter().cloned() {
            let geometry = self.scene.geometry(geometry_ref).unwrap();
            let geometry_data = self.geometry_data.get(geometry_ref.0 as usize).unwrap();

            let position_buffer = resource_loader.get_buffer(geometry_data.position_buffer)?;
            let index_buffer = resource_loader.get_buffer(geometry_data.index_buffer)?;

            let position_buffer_address = unsafe { context.device.get_buffer_device_address_helper(position_buffer) };
            let index_buffer_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

            accel_geometry.push(vk::AccelerationStructureGeometryKHR {
                geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                geometry: vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        vertex_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: position_buffer_address,
                        },
                        vertex_stride: mem::size_of::<PositionData>() as vk::DeviceSize,
                        max_vertex: geometry.positions.len() as u32,
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
            let triangle_count = geometry.indices.len() as u32;
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
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                ..Default::default()
            };
            unsafe { context.device.create_acceleration_structure_khr(&create_info, None) }.unwrap()
        };

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

        schedule.add_compute(
            command_name!("build"),
            |params| {
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

    pub fn update<'a>(
        &mut self,
        context: &'a Arc<Context>,
        resource_loader: &ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) {
        while self.bottom_level_accel.len() < self.clusters.len() {
            if let Some(accel) = self.create_bottom_level_accel(
                self.clusters.get(self.bottom_level_accel.len()).unwrap(),
                context,
                resource_loader,
                global_allocator,
                schedule,
            ) {
                self.bottom_level_accel.push(accel);
            } else {
                break;
            }
        }
    }
}
