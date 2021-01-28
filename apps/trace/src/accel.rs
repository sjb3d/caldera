use crate::scene::*;
use caldera::*;
use caldera_macro::descriptor_set_layout;
use spark::{vk, Builder};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::{mem, slice};

#[repr(C)]
struct PositionData([f32; 3]);

#[repr(C)]
struct IndexData([u32; 3]);

#[repr(C)]
pub struct CameraData {
    ray_origin: [f32; 3],
    ray_vec_from_coord: [f32; 9],
}

descriptor_set_layout!(pub DebugDescriptorSetLayout {
    camera: UniformData<CameraData>,
    accel: AccelerationStructure,
    output: StorageImage,
});

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
    accel_device_address: AtomicU64,
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

impl SceneShared {
    fn cluster_instance_count(&self) -> usize {
        self.clusters.iter().map(|cluster| cluster.transform_refs.len()).sum()
    }
}

#[derive(Clone, Copy)]
struct ShaderBindingRegion {
    offset: u32,
    stride: u32,
    size: u32,
}

impl ShaderBindingRegion {
    fn into_device_address_region(&self, base_device_address: vk::DeviceAddress) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_device_address + self.offset as vk::DeviceSize,
            stride: self.stride as vk::DeviceSize,
            size: self.size as vk::DeviceSize,
        }
    }
}

pub struct SceneAccel {
    shared: Arc<SceneShared>,
    context: Arc<Context>,
    debug_descriptor_set_layout: DebugDescriptorSetLayout,
    debug_pipeline_layout: vk::PipelineLayout,
    debug_pipeline: vk::Pipeline,
    shader_binding_table: StaticBufferHandle,
    shader_binding_raygen_region: ShaderBindingRegion,
    shader_binding_miss_region: ShaderBindingRegion,
    shader_binding_hit_region: ShaderBindingRegion,
    geometry_data: Vec<TriangleMeshData>,
    bottom_level_accel: Vec<BottomLevelAccel>,
    instance_buffer: Option<StaticBufferHandle>,
    top_level_accel: Option<TopLevelAccel>,
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
                accel_device_address: AtomicU64::new(0),
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

    pub fn new(
        scene: Scene,
        context: &Arc<Context>,
        descriptor_set_layout_cache: &mut DescriptorSetLayoutCache,
        pipeline_cache: &PipelineCache,
        resource_loader: &mut ResourceLoader,
    ) -> Self {
        let clusters = Self::make_clusters(&scene);
        let shared = Arc::new(SceneShared { scene, clusters });

        // make pipeline
        let debug_descriptor_set_layout = DebugDescriptorSetLayout::new(descriptor_set_layout_cache);
        let debug_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(debug_descriptor_set_layout.0);

        let debug_pipeline = pipeline_cache.get_ray_tracing(
            "trace/debug.rgen.spv",
            "trace/debug.rchit.spv",
            "trace/debug.rmiss.spv",
            debug_pipeline_layout,
        );

        // make shader binding table layout
        let (
            shader_binding_raygen_region,
            shader_binding_miss_region,
            shader_binding_hit_region,
            shader_binding_table_size,
        ) = {
            let rtpp = context.ray_tracing_pipeline_properties.as_ref().unwrap();

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

            let miss_record_size = 0;
            let miss_stride = align_up(
                rtpp.shader_group_handle_size + miss_record_size,
                rtpp.shader_group_handle_alignment,
            );
            let miss_region = ShaderBindingRegion {
                offset: next_offset,
                stride: miss_stride,
                size: miss_stride,
            };
            next_offset += align_up(miss_region.size, rtpp.shader_group_base_alignment);

            let hit_record_size = 0;
            let hit_stride = align_up(
                rtpp.shader_group_handle_size + hit_record_size,
                rtpp.shader_group_handle_alignment,
            );
            let hit_region = ShaderBindingRegion {
                offset: next_offset,
                stride: hit_stride,
                size: hit_stride,
            };
            next_offset += align_up(hit_region.size, rtpp.shader_group_base_alignment);

            (raygen_region, miss_region, hit_region, next_offset)
        };

        // write the table
        let shader_binding_table = resource_loader.create_buffer();
        resource_loader.async_load({
            let context = Arc::clone(context);
            move |allocator| {
                let rtpp = context.ray_tracing_pipeline_properties.as_ref().unwrap();
                let shader_group_count = 3;
                let handle_size = rtpp.shader_group_handle_size as usize;
                let mut handle_data = vec![0u8; (shader_group_count as usize) * handle_size];
                unsafe {
                    context.device.get_ray_tracing_shader_group_handles_khr(
                        debug_pipeline,
                        0,
                        shader_group_count,
                        &mut handle_data,
                    )
                }
                .unwrap();

                let remain = handle_data.as_slice();
                let (raygen_group_handle, remain) = remain.split_at(handle_size);
                let (miss_group_handle, remain) = remain.split_at(handle_size);
                let hit_group_handle = remain;

                let desc = BufferDesc::new(shader_binding_table_size as usize);
                let mut writer = allocator
                    .map_buffer(
                        shader_binding_table,
                        &desc,
                        BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                    )
                    .unwrap();

                assert_eq!(shader_binding_raygen_region.offset, 0);
                writer.write_all(raygen_group_handle);

                writer.write_zeros(shader_binding_miss_region.offset as usize - writer.written());
                writer.write_all(miss_group_handle);

                writer.write_zeros(shader_binding_hit_region.offset as usize - writer.written());
                writer.write_all(hit_group_handle);
            }
        });

        // make vertex/index buffers for each geometry
        let geometry_data = shared
            .scene
            .geometry_ref_iter()
            .map(|geometry_ref| {
                let data = TriangleMeshData {
                    position_buffer: resource_loader.create_buffer(),
                    index_buffer: resource_loader.create_buffer(),
                };
                resource_loader.async_load({
                    let shared = Arc::clone(&shared);
                    move |allocator| {
                        let geometry = shared.scene.geometry(geometry_ref).unwrap();
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

        Self {
            shared,
            context: Arc::clone(&context),
            debug_descriptor_set_layout,
            debug_pipeline_layout,
            debug_pipeline,
            shader_binding_table,
            shader_binding_raygen_region,
            shader_binding_miss_region,
            shader_binding_hit_region,
            geometry_data,
            bottom_level_accel: Vec::new(),
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
        for geometry_ref in cluster.geometry_refs.iter().cloned() {
            let geometry = self.shared.scene.geometry(geometry_ref).unwrap();
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

        let accel_device_address = {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR {
                acceleration_structure: Some(accel),
                ..Default::default()
            };
            unsafe { context.device.get_acceleration_structure_device_address_khr(&info) }
        };
        cluster
            .accel_device_address
            .store(accel_device_address, Ordering::SeqCst);

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

    pub fn create_instance_buffer<'a>(&self, resource_loader: &mut ResourceLoader) -> StaticBufferHandle {
        let instance_buffer = resource_loader.create_buffer();
        resource_loader.async_load({
            let shared = Arc::clone(&self.shared);
            move |allocator| {
                let count = shared.cluster_instance_count();
                println!("instance count: {}", count);

                let desc = BufferDesc::new(count * mem::size_of::<vk::AccelerationStructureInstanceKHR>());
                let mut writer = allocator
                    .map_buffer(instance_buffer, &desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                    .unwrap();

                for cluster in shared.clusters.iter() {
                    let acceleration_structure_reference = cluster.accel_device_address.load(Ordering::SeqCst);
                    for transform_ref in cluster.transform_refs.iter().cloned() {
                        let custom_index = transform_ref.0 & 0x00_ff_ff_ff;
                        let transform = shared.scene.transform(transform_ref).unwrap();
                        let instance = vk::AccelerationStructureInstanceKHR {
                            transform: vk::TransformMatrixKHR {
                                matrix: transform.0.into_transposed_transform(),
                            },
                            instance_custom_index_and_mask: 0xff_00_00_00 | custom_index,
                            instance_shader_binding_table_record_offset_and_flags: 0, // TODO
                            acceleration_structure_reference,
                        };
                        writer.write_all(instance.as_byte_slice());
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

        let instance_count = self.shared.cluster_instance_count() as u32;
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
                for bottom_level_accel in self.bottom_level_accel.iter() {
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
        while self.bottom_level_accel.len() < self.shared.clusters.len() {
            if let Some(accel) = self.create_bottom_level_accel(
                self.shared.clusters.get(self.bottom_level_accel.len()).unwrap(),
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

        // make instance buffer from transforms
        if self.instance_buffer.is_none() && self.bottom_level_accel.len() == self.shared.clusters.len() {
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

    pub fn trace<'a>(
        &'a self,
        context: &'a Context,
        resource_loader: &ResourceLoader,
        schedule: &mut RenderSchedule<'a>,
        descriptor_pool: &'a DescriptorPool,
        swap_extent: &'a vk::Extent2D,
        ray_origin: Vec3,
        ray_vec_from_coord: Mat3,
    ) -> Option<ImageHandle> {
        let top_level_accel = self.top_level_accel.as_ref()?;
        let shader_binding_table_buffer = resource_loader.get_buffer(self.shader_binding_table)?;

        let shader_binding_table_address = unsafe {
            context
                .device
                .get_buffer_device_address_helper(shader_binding_table_buffer)
        };

        let output_desc = ImageDesc::new_2d(
            swap_extent.width,
            swap_extent.height,
            vk::Format::R32_UINT,
            vk::ImageAspectFlags::COLOR,
        );
        let output_image = schedule.describe_image(&output_desc);

        schedule.add_compute(
            command_name!("trace"),
            |params| {
                for bottom_level_accel in self.bottom_level_accel.iter() {
                    params.add_buffer(
                        bottom_level_accel.buffer,
                        BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
                    );
                }
                params.add_buffer(top_level_accel.buffer, BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE);
                params.add_image(output_image, ImageUsage::RAY_TRACING_STORAGE_WRITE);
            },
            move |params, cmd| {
                let output_image_view = params.get_image_view(output_image);

                let debug_descriptor_set = self.debug_descriptor_set_layout.write(
                    &descriptor_pool,
                    &|buf: &mut CameraData| {
                        *buf = CameraData {
                            ray_origin: ray_origin.into(),
                            ray_vec_from_coord: *ray_vec_from_coord.as_array(),
                        }
                    },
                    top_level_accel.accel,
                    output_image_view,
                );

                unsafe {
                    context
                        .device
                        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, self.debug_pipeline);
                    context.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        self.debug_pipeline_layout,
                        0,
                        slice::from_ref(&debug_descriptor_set),
                        &[],
                    );
                }

                let raygen_shader_binding_table = self
                    .shader_binding_raygen_region
                    .into_device_address_region(shader_binding_table_address);
                let miss_shader_binding_table = self
                    .shader_binding_miss_region
                    .into_device_address_region(shader_binding_table_address);
                let hit_shader_binding_table = self
                    .shader_binding_hit_region
                    .into_device_address_region(shader_binding_table_address);
                let callable_shader_binding_table = vk::StridedDeviceAddressRegionKHR::default();
                unsafe {
                    context.device.cmd_trace_rays_khr(
                        cmd,
                        &raygen_shader_binding_table,
                        &miss_shader_binding_table,
                        &hit_shader_binding_table,
                        &callable_shader_binding_table,
                        swap_extent.width,
                        swap_extent.height,
                        1,
                    );
                }
            },
        );

        Some(output_image)
    }
}

impl Drop for SceneAccel {
    fn drop(&mut self) {
        let device = &self.context.device;
        for bottom_level_accel in self.bottom_level_accel.iter() {
            unsafe { device.destroy_acceleration_structure_khr(Some(bottom_level_accel.accel), None) };
        }
        if let Some(top_level_accel) = self.top_level_accel.as_ref() {
            unsafe { device.destroy_acceleration_structure_khr(Some(top_level_accel.accel), None) };
        }
    }
}
