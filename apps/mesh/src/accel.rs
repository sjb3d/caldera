use crate::loader::*;
use bytemuck::{Pod, Zeroable};
use caldera::*;
use spark::vk;
use std::sync::Arc;
use std::{mem, slice};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct TraceData {
    ray_origin: Vec3,
    ray_vec_from_coord: Mat3,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct HitRecordData {
    index_buffer_address: u64,
    attribute_buffer_address: u64,
}

descriptor_set_layout!(TraceDescriptorSetLayout {
    trace: UniformData<TraceData>,
    accel: AccelerationStructure,
    output: StorageImage,
});

// vk::AccelerationStructureInstanceKHR with Pod trait
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct AccelerationStructureInstance {
    transform: TransposedTransform3,
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

struct AccelLevel {
    context: Arc<Context>,
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

impl AccelLevel {
    fn new_bottom_level<'a>(
        context: &'a Arc<Context>,
        mesh_info: &MeshInfo,
        resource_loader: &ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> AccelLevel {
        let position_buffer = resource_loader.get_buffer(mesh_info.position_buffer).unwrap();
        let index_buffer = resource_loader.get_buffer(mesh_info.index_buffer).unwrap();

        let vertex_buffer_address = unsafe { context.device.get_buffer_device_address_helper(position_buffer) };
        let index_buffer_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

        let geometry_triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR {
            vertex_format: vk::Format::R32G32B32_SFLOAT,
            vertex_data: vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_buffer_address,
            },
            vertex_stride: mem::size_of::<PositionData>() as vk::DeviceSize,
            max_vertex: mesh_info.vertex_count - 1,
            index_type: vk::IndexType::UINT32,
            index_data: vk::DeviceOrHostAddressConstKHR {
                device_address: index_buffer_address,
            },
            ..Default::default()
        };

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::TRIANGLES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                triangles: geometry_triangles_data,
            },
            flags: vk::GeometryFlagsKHR::OPAQUE,
            ..Default::default()
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries: &geometry,
            ..Default::default()
        };

        let sizes = {
            let max_primitive_count = mesh_info.triangle_count;
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                context.device.get_acceleration_structure_build_sizes_khr(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    Some(slice::from_ref(&max_primitive_count)),
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

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as usize));

        schedule.add_compute(
            command_name!("build"),
            |params| {
                params.add_buffer(buffer, BufferUsage::ACCELERATION_STRUCTURE_WRITE);
                params.add_buffer(scratch_buffer, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
            },
            {
                let triangle_count = mesh_info.triangle_count;
                move |params, cmd| {
                    let scratch_buffer = params.get_buffer(scratch_buffer);

                    let scratch_buffer_address =
                        unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
                        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                        dst_acceleration_structure: Some(accel),
                        geometry_count: 1,
                        p_geometries: &geometry,
                        scratch_data: vk::DeviceOrHostAddressKHR {
                            device_address: scratch_buffer_address,
                        },
                        ..Default::default()
                    };

                    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: triangle_count,
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
                }
            },
        );

        Self {
            context: Arc::clone(&context),
            accel,
            buffer,
        }
    }

    fn new_top_level<'a>(
        context: &'a Arc<Context>,
        bottom_level_buffer: BufferHandle,
        instance_buffer: vk::Buffer,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> Self {
        let instance_buffer_address = unsafe { context.device.get_buffer_device_address_helper(instance_buffer) };

        let geometry_instance_data = vk::AccelerationStructureGeometryInstancesDataKHR {
            data: vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer_address,
            },
            ..Default::default()
        };

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: geometry_instance_data,
            },
            ..Default::default()
        };

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries: &geometry,
            ..Default::default()
        };

        let instance_count = MeshInfo::INSTANCE_COUNT as u32;
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
                params.add_buffer(bottom_level_buffer, BufferUsage::ACCELERATION_STRUCTURE_READ);
                params.add_buffer(buffer, BufferUsage::ACCELERATION_STRUCTURE_WRITE);
                params.add_buffer(scratch_buffer, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
            },
            move |params, cmd| {
                let scratch_buffer = params.get_buffer(scratch_buffer);

                let scratch_buffer_address = unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

                let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                    flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    dst_acceleration_structure: Some(accel),
                    geometry_count: 1,
                    p_geometries: &geometry,
                    scratch_data: vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffer_address,
                    },
                    ..Default::default()
                };

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

        Self {
            context: Arc::clone(&context),
            accel,
            buffer,
        }
    }
}

impl Drop for AccelLevel {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.destroy_acceleration_structure_khr(Some(self.accel), None) };
    }
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

pub struct AccelInfo {
    trace_descriptor_set_layout: TraceDescriptorSetLayout,
    trace_pipeline_layout: vk::PipelineLayout,
    trace_pipeline: vk::Pipeline,
    shader_binding_table: StaticBufferHandle,
    shader_binding_raygen_region: ShaderBindingRegion,
    shader_binding_miss_region: ShaderBindingRegion,
    shader_binding_hit_region: ShaderBindingRegion,
    bottom_level: AccelLevel,
    instance_buffer: StaticBufferHandle,
    top_level: Option<AccelLevel>,
}

impl AccelInfo {
    pub fn new<'a>(
        context: &'a Arc<Context>,
        descriptor_set_layout_cache: &mut DescriptorSetLayoutCache,
        pipeline_cache: &PipelineCache,
        resource_loader: &mut ResourceLoader,
        mesh_info: &'a MeshInfo,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> Self {
        let trace_descriptor_set_layout = TraceDescriptorSetLayout::new(descriptor_set_layout_cache);
        let trace_pipeline_layout = descriptor_set_layout_cache.create_pipeline_layout(trace_descriptor_set_layout.0);

        let index_buffer = resource_loader.get_buffer(mesh_info.index_buffer).unwrap();
        let index_buffer_device_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

        let attribute_buffer = resource_loader.get_buffer(mesh_info.attribute_buffer).unwrap();
        let attribute_buffer_device_address =
            unsafe { context.device.get_buffer_device_address_helper(attribute_buffer) };

        // TODO: figure out live reload, needs to regenerate SBT!
        let trace_pipeline = pipeline_cache.get_ray_tracing(
            &[
                RayTracingShaderGroupDesc::Raygen("mesh/trace.rgen.spv"),
                RayTracingShaderGroupDesc::Miss("mesh/trace.rmiss.spv"),
                RayTracingShaderGroupDesc::Hit {
                    closest_hit: "mesh/trace.rchit.spv",
                    any_hit: None,
                    intersection: None,
                },
            ],
            trace_pipeline_layout,
        );

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

            let hit_record_size = mem::size_of::<HitRecordData>() as u32;
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
                        trace_pipeline,
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

                let hit_data = HitRecordData {
                    index_buffer_address: index_buffer_device_address,
                    attribute_buffer_address: attribute_buffer_device_address,
                };

                let desc = BufferDesc::new(shader_binding_table_size as usize);
                let mut writer = allocator
                    .map_buffer(
                        shader_binding_table,
                        &desc,
                        BufferUsage::RAY_TRACING_SHADER_BINDING_TABLE,
                    )
                    .unwrap();

                assert_eq!(shader_binding_raygen_region.offset, 0);
                writer.write(raygen_group_handle);

                writer.write_zeros(shader_binding_miss_region.offset as usize - writer.written());
                writer.write(miss_group_handle);

                writer.write_zeros(shader_binding_hit_region.offset as usize - writer.written());
                writer.write(hit_group_handle);
                writer.write(&hit_data);
            }
        });

        let bottom_level =
            AccelLevel::new_bottom_level(context, mesh_info, resource_loader, global_allocator, schedule);

        let bottom_level_device_address = {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR {
                acceleration_structure: Some(bottom_level.accel),
                ..Default::default()
            };
            unsafe { context.device.get_acceleration_structure_device_address_khr(&info) }
        };

        let instance_buffer = resource_loader.create_buffer();
        resource_loader.async_load({
            let instances = mesh_info.instances;
            move |allocator| {
                let desc =
                    BufferDesc::new(MeshInfo::INSTANCE_COUNT * mem::size_of::<vk::AccelerationStructureInstanceKHR>());
                let mut writer = allocator
                    .map_buffer(instance_buffer, &desc, BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT)
                    .unwrap();

                for src in instances.iter() {
                    let instance = AccelerationStructureInstance {
                        transform: src.into_transform().transposed(),
                        instance_custom_index_and_mask: 0xff_00_00_00,
                        instance_shader_binding_table_record_offset_and_flags: 0,
                        acceleration_structure_reference: bottom_level_device_address,
                    };
                    writer.write(&instance);
                }
            }
        });

        Self {
            trace_descriptor_set_layout,
            trace_pipeline_layout,
            trace_pipeline,
            shader_binding_table,
            shader_binding_raygen_region,
            shader_binding_miss_region,
            shader_binding_hit_region,
            bottom_level,
            instance_buffer,
            top_level: None,
        }
    }

    pub fn update<'a>(
        &mut self,
        context: &'a Arc<Context>,
        resource_loader: &ResourceLoader,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) {
        if self.top_level.is_none() {
            if let Some(instance_buffer) = resource_loader.get_buffer(self.instance_buffer) {
                self.top_level = Some(AccelLevel::new_top_level(
                    context,
                    self.bottom_level.buffer,
                    instance_buffer,
                    global_allocator,
                    schedule,
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch<'a>(
        &'a self,
        context: &'a Context,
        resource_loader: &ResourceLoader,
        schedule: &mut RenderSchedule<'a>,
        descriptor_pool: &'a DescriptorPool,
        swap_size: UVec2,
        ray_origin: Vec3,
        ray_vec_from_coord: Mat3,
    ) -> Option<ImageHandle> {
        let top_level = self.top_level.as_ref()?;
        let shader_binding_table_buffer = resource_loader.get_buffer(self.shader_binding_table)?;

        let shader_binding_table_address = unsafe {
            context
                .device
                .get_buffer_device_address_helper(shader_binding_table_buffer)
        };

        let output_desc = ImageDesc::new_2d(swap_size, vk::Format::R32_UINT, vk::ImageAspectFlags::COLOR);
        let output_image = schedule.describe_image(&output_desc);

        schedule.add_compute(
            command_name!("trace"),
            |params| {
                params.add_buffer(
                    self.bottom_level.buffer,
                    BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE,
                );
                params.add_buffer(top_level.buffer, BufferUsage::RAY_TRACING_ACCELERATION_STRUCTURE);
                params.add_image(output_image, ImageUsage::RAY_TRACING_STORAGE_WRITE);
            },
            move |params, cmd| {
                let output_image_view = params.get_image_view(output_image);

                let trace_descriptor_set = self.trace_descriptor_set_layout.write(
                    &descriptor_pool,
                    |buf: &mut TraceData| {
                        *buf = TraceData {
                            ray_origin,
                            ray_vec_from_coord,
                        }
                    },
                    top_level.accel,
                    output_image_view,
                );

                unsafe {
                    context
                        .device
                        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, self.trace_pipeline);
                    context.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        self.trace_pipeline_layout,
                        0,
                        slice::from_ref(&trace_descriptor_set),
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
                        swap_size.x,
                        swap_size.y,
                        1,
                    );
                }
            },
        );

        Some(output_image)
    }
}
