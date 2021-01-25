use caldera::*;
use caldera_macro::descriptor_set_layout;
use imgui::im_str;
use imgui::Key;
use ply_rs::{parser, ply};
use spark::vk;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};
use std::{env, fs, io, mem, slice};
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[derive(Clone, Copy)]
struct PlyVertex {
    pos: Vec3,
}

#[derive(Clone, Copy)]
struct PlyFace {
    indices: UVec3,
}

#[repr(C)]
struct PositionData {
    pos: [f32; 3],
}

#[repr(C)]
struct AttributeData {
    normal: [f32; 3],
}

#[repr(C)]
struct InstanceData {
    matrix: [f32; 12],
}

#[repr(C)]
struct TestData {
    proj_from_world: [f32; 16],
}

descriptor_set_layout!(TestDescriptorSetLayout {
    test: UniformData<TestData>,
});

#[repr(C)]
struct TraceData {
    ray_origin: [f32; 3],
    ray_vec_from_coord: [f32; 9],
}

#[repr(C)]
struct HitRecordData {
    index_buffer_address: u64,
    attribute_buffer_address: u64,
}

descriptor_set_layout!(TraceDescriptorSetLayout {
    trace: UniformData<TraceData>,
    accel: AccelerationStructure,
    output: StorageImage,
});

descriptor_set_layout!(CopyDescriptorSetLayout { ids: StorageImage });

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self { pos: Vec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos.x = v,
            ("y", ply::Property::Float(v)) => self.pos.y = v,
            ("z", ply::Property::Float(v)) => self.pos.z = v,
            _ => {}
        }
    }
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        Self { indices: UVec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(v)) => {
                assert_eq!(v.len(), 3);
                for (dst, src) in self.indices.as_mut_slice().iter_mut().zip(v.iter().rev()) {
                    *dst = *src as u32;
                }
            }
            (k, _) => panic!("unknown key {}", k),
        }
    }
}

#[derive(Clone, Copy)]
struct MeshInfo {
    position_buffer: StaticBufferHandle,
    attribute_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
    instances: [Similarity3; Self::INSTANCE_COUNT],
    instance_buffer: StaticBufferHandle,
    vertex_count: u32,
    triangle_count: u32,
}

impl MeshInfo {
    const INSTANCE_COUNT: usize = 8;

    fn new(resource_loader: &mut ResourceLoader) -> Self {
        Self {
            position_buffer: resource_loader.create_buffer(),
            attribute_buffer: resource_loader.create_buffer(),
            index_buffer: resource_loader.create_buffer(),
            instances: [Similarity3::identity(); Self::INSTANCE_COUNT],
            instance_buffer: resource_loader.create_buffer(),
            vertex_count: 0,
            triangle_count: 0,
        }
    }

    fn load(&mut self, allocator: &mut ResourceAllocator, mesh_file_name: &str, with_ray_tracing: bool) {
        let vertex_parser = parser::Parser::<PlyVertex>::new();
        let face_parser = parser::Parser::<PlyFace>::new();

        let mut f = io::BufReader::new(fs::File::open(mesh_file_name).unwrap());
        let header = vertex_parser.read_header(&mut f).unwrap();

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for (_key, element) in header.elements.iter() {
            match element.name.as_ref() {
                "vertex" => {
                    vertices = vertex_parser
                        .read_payload_for_element(&mut f, element, &header)
                        .unwrap();
                }
                "face" => {
                    faces = face_parser.read_payload_for_element(&mut f, element, &header).unwrap();
                }
                _ => panic!("unexpected element {:?}", element),
            }
        }

        let position_buffer_desc = BufferDesc::new((vertices.len() * mem::size_of::<PositionData>()) as u32);
        let mut mapping = allocator
            .map_buffer::<PositionData>(
                self.position_buffer,
                &position_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::VERTEX_BUFFER | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                } else {
                    BufferUsage::VERTEX_BUFFER
                },
            )
            .unwrap();
        let mut min = Vec3::broadcast(f32::MAX);
        let mut max = Vec3::broadcast(-f32::MAX);
        for (dst, src) in mapping.get_mut().iter_mut().zip(vertices.iter()) {
            let v = src.pos;
            dst.pos = *v.as_array();
            min = min.min_by_component(v);
            max = max.max_by_component(v);
        }
        self.vertex_count = vertices.len() as u32;

        let index_buffer_desc = BufferDesc::new((faces.len() * 3 * mem::size_of::<u32>()) as u32);
        let mut mapping = allocator
            .map_buffer::<u32>(
                self.index_buffer,
                &index_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::INDEX_BUFFER | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                } else {
                    BufferUsage::INDEX_BUFFER
                },
            )
            .unwrap();
        let mut normals = vec![Vec3::zero(); vertices.len()];
        for (dst, src) in mapping.get_mut().chunks_mut(3).zip(faces.iter()) {
            for i in 0..3 {
                dst[i] = src.indices[i];
            }
            let v0 = vertices[src.indices[0] as usize].pos;
            let v1 = vertices[src.indices[1] as usize].pos;
            let v2 = vertices[src.indices[2] as usize].pos;
            let normal = (v2 - v0).cross(v1 - v0).normalized();
            if !normal.x.is_nan() && !normal.y.is_nan() && !normal.z.is_nan() {
                // TODO: weight by angle at vertex?
                normals[src.indices[0] as usize] += normal;
                normals[src.indices[1] as usize] += normal;
                normals[src.indices[2] as usize] += normal;
            }
        }
        self.triangle_count = faces.len() as u32;

        let attribute_buffer_desc = BufferDesc::new((vertices.len() * mem::size_of::<AttributeData>()) as u32);
        let mut mapping = allocator
            .map_buffer::<AttributeData>(
                self.attribute_buffer,
                &attribute_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::VERTEX_BUFFER | BufferUsage::RAY_TRACING_STORAGE_READ
                } else {
                    BufferUsage::VERTEX_BUFFER
                },
            )
            .unwrap();
        for (dst, src) in mapping.get_mut().iter_mut().zip(normals.iter()) {
            dst.normal = *src.as_array();
        }

        let scale = 0.9 / (max - min).component_max();
        let offset = (-0.5 * scale) * (max + min);
        for i in 0..Self::INSTANCE_COUNT {
            let corner = |i: usize, b| if ((i >> b) & 1usize) != 0usize { 0.5 } else { -0.5 };
            self.instances[i] = Similarity3::new(
                offset + Vec3::new(corner(i, 0), corner(i, 1), corner(i, 2)),
                Rotor3::identity(),
                scale,
            );
        }

        let instance_buffer_desc = BufferDesc::new((Self::INSTANCE_COUNT * mem::size_of::<InstanceData>()) as u32);
        let mut mapping = allocator
            .map_buffer::<InstanceData>(self.instance_buffer, &instance_buffer_desc, BufferUsage::VERTEX_BUFFER)
            .unwrap();
        for (dst, src) in mapping.get_mut().iter_mut().zip(self.instances.iter()) {
            *dst = InstanceData {
                matrix: src.into_transposed_transform(),
            };
        }
    }
}

struct AccelLevel {
    accel: vk::AccelerationStructureKHR,
    buffer: BufferHandle,
}

impl AccelLevel {
    fn new_bottom_level<'a>(
        context: &'a Arc<Context>,
        mesh_info: &'a MeshInfo,
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
            vertex_stride: 12,
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

        let buffer_desc = BufferDesc::new(sizes.acceleration_structure_size as u32);
        let buffer = schedule.create_buffer(
            &buffer_desc,
            BufferUsage::ACCELERATION_STRUCTURE_WRITE | BufferUsage::ACCELERATION_STRUCTURE_READ,
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

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as u32));

        schedule.add_compute(
            command_name!("build"),
            |params| {
                params.add_buffer(scratch_buffer, BufferUsage::ACCELERATION_STRUCTURE_BUILD_SCRATCH);
            },
            move |params, cmd| {
                let scratch_buffer = params.get_buffer(scratch_buffer);

                let scratch_buffer_address = unsafe { context.device.get_buffer_device_address_helper(scratch_buffer) };

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
                    primitive_count: mesh_info.triangle_count,
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

        Self { accel, buffer }
    }

    fn new_top_level<'a>(
        context: &'a Arc<Context>,
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

        let buffer_desc = BufferDesc::new(sizes.acceleration_structure_size as u32);
        let buffer = schedule.create_buffer(
            &buffer_desc,
            BufferUsage::ACCELERATION_STRUCTURE_WRITE | BufferUsage::ACCELERATION_STRUCTURE_READ,
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

        let scratch_buffer = schedule.describe_buffer(&BufferDesc::new(sizes.build_scratch_size as u32));

        schedule.add_compute(
            command_name!("build"),
            |params| {
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

        Self { accel, buffer }
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

struct AccelInfo {
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
    fn new<'a>(
        context: &'a Arc<Context>,
        pipeline_cache: &PipelineCache,
        descriptor_pool: &DescriptorPool,
        resource_loader: &mut ResourceLoader,
        mesh_info: &'a MeshInfo,
        global_allocator: &mut Allocator,
        schedule: &mut RenderSchedule<'a>,
    ) -> Self {
        let trace_descriptor_set_layout = TraceDescriptorSetLayout::new(&descriptor_pool);
        let trace_pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout_from_ref(&trace_descriptor_set_layout.0)
        }
        .unwrap();

        let index_buffer = resource_loader.get_buffer(mesh_info.index_buffer).unwrap();
        let index_buffer_device_address = unsafe { context.device.get_buffer_device_address_helper(index_buffer) };

        let attribute_buffer = resource_loader.get_buffer(mesh_info.attribute_buffer).unwrap();
        let attribute_buffer_device_address =
            unsafe { context.device.get_buffer_device_address_helper(attribute_buffer) };

        // TODO: figure out live reload, needs to regenerate SBT!
        let trace_pipeline = pipeline_cache.get_ray_tracing(
            "mesh/trace.rgen.spv",
            "mesh/trace.rchit.spv",
            "mesh/trace.rmiss.spv",
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

                let desc = BufferDesc::new(shader_binding_table_size);
                let mut mapping = allocator
                    .map_buffer::<u8>(shader_binding_table, &desc, BufferUsage::SHADER_BINDING_TABLE)
                    .unwrap();

                let remain = mapping.get_mut();
                let (remain, hit_mapping) = remain.split_at_mut(shader_binding_hit_region.offset as usize);
                let (remain, miss_mapping) = remain.split_at_mut(shader_binding_miss_region.offset as usize);
                let raygen_mapping = remain;

                let hit_data = HitRecordData {
                    index_buffer_address: index_buffer_device_address,
                    attribute_buffer_address: attribute_buffer_device_address,
                };
                let hit_data_bytes: [u8; 16] = unsafe { mem::transmute(hit_data) };

                raygen_mapping[..handle_size].copy_from_slice(raygen_group_handle);
                miss_mapping[..handle_size].copy_from_slice(miss_group_handle);
                hit_mapping[..handle_size].copy_from_slice(hit_group_handle);
                hit_mapping[handle_size..handle_size + 16].copy_from_slice(&hit_data_bytes);
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
                let desc = BufferDesc::new(
                    (MeshInfo::INSTANCE_COUNT * mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u32,
                );
                let mut mapping = allocator
                    .map_buffer::<vk::AccelerationStructureInstanceKHR>(
                        instance_buffer,
                        &desc,
                        BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT,
                    )
                    .unwrap();

                for (dst, src) in mapping.get_mut().iter_mut().zip(instances.iter()) {
                    *dst = vk::AccelerationStructureInstanceKHR {
                        transform: vk::TransformMatrixKHR {
                            matrix: src.into_transposed_transform(),
                        },
                        instance_custom_index_and_mask: 0xff_00_00_00,
                        instance_shader_binding_table_record_offset_and_flags: 0,
                        acceleration_structure_reference: bottom_level_device_address,
                    };
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

    fn update<'a>(
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
                    instance_buffer,
                    global_allocator,
                    schedule,
                ));
            }
        }
    }

    fn dispatch<'a>(
        &'a self,
        context: &'a Arc<Context>,
        resource_loader: &ResourceLoader,
        schedule: &mut RenderSchedule<'a>,
        descriptor_pool: &'a DescriptorPool,
        swap_extent: &'a vk::Extent2D,
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
                params.add_buffer(self.bottom_level.buffer, BufferUsage::ACCELERATION_STRUCTURE_READ);
                params.add_image(output_image, ImageUsage::RAY_TRACING_STORAGE_WRITE);
            },
            move |params, cmd| {
                let output_image_view = params.get_image_view(output_image);

                let trace_descriptor_set = self.trace_descriptor_set_layout.write(
                    &descriptor_pool,
                    &|buf: &mut TraceData| {
                        *buf = TraceData {
                            ray_origin: ray_origin.into(),
                            ray_vec_from_coord: *ray_vec_from_coord.as_array(),
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

#[derive(Clone, Copy, Eq, PartialEq)]
enum RenderMode {
    Raster,
    RasterMultisampled,
    RayTrace,
}

struct App {
    context: Arc<Context>,

    test_descriptor_set_layout: TestDescriptorSetLayout,
    test_pipeline_layout: vk::PipelineLayout,

    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    mesh_info: Arc<Mutex<MeshInfo>>,
    accel_info: Option<AccelInfo>,
    render_mode: RenderMode,
    is_rotating: bool,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: String) -> Self {
        let context = &base.context;
        let descriptor_pool = &base.systems.descriptor_pool;

        let test_descriptor_set_layout = TestDescriptorSetLayout::new(&descriptor_pool);
        let test_pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout_from_ref(&test_descriptor_set_layout.0)
        }
        .unwrap();

        let copy_descriptor_set_layout = CopyDescriptorSetLayout::new(&descriptor_pool);
        let copy_pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout_from_ref(&copy_descriptor_set_layout.0)
        }
        .unwrap();

        let mesh_info = Arc::new(Mutex::new(MeshInfo::new(&mut base.systems.resource_loader)));
        base.systems.resource_loader.async_load({
            let mesh_info = Arc::clone(&mesh_info);
            let with_ray_tracing = context.device.extensions.khr_acceleration_structure;
            move |allocator| {
                let mut mesh_info_clone = *mesh_info.lock().unwrap();
                mesh_info_clone.load(allocator, &mesh_file_name, with_ray_tracing);
                *mesh_info.lock().unwrap() = mesh_info_clone;
            }
        });

        Self {
            context: Arc::clone(&context),
            test_descriptor_set_layout,
            test_pipeline_layout,
            copy_descriptor_set_layout,
            copy_pipeline_layout,
            mesh_info,
            accel_info: None,
            render_mode: RenderMode::Raster,
            is_rotating: false,
            angle: PI / 8.0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(ui.key_index(Key::Escape)) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                let ui = &ui;
                let context = &base.context;
                let render_mode = &mut self.render_mode;
                let is_rotating = &mut self.is_rotating;
                move || {
                    ui.text("Render Mode:");
                    ui.radio_button(im_str!("Raster"), render_mode, RenderMode::Raster);
                    ui.radio_button(
                        im_str!("Raster (Multisampled)"),
                        render_mode,
                        RenderMode::RasterMultisampled,
                    );
                    if context.device.extensions.khr_acceleration_structure {
                        ui.radio_button(im_str!("Ray Trace"), render_mode, RenderMode::RayTrace);
                    } else {
                        ui.text_disabled("Ray Tracing Not Supported!");
                    }
                    ui.checkbox(im_str!("Rotate"), is_rotating);
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore);
        let swap_extent = base.display.swapchain.get_extent();
        let swap_format = base.display.swapchain.get_format();
        let swap_image = schedule.import_image(
            &ImageDesc::new_2d(
                swap_extent.width,
                swap_extent.height,
                swap_format,
                vk::ImageAspectFlags::COLOR,
            ),
            ImageUsage::COLOR_ATTACHMENT_WRITE | ImageUsage::SWAPCHAIN,
            swap_vk_image,
            ImageUsage::empty(),
        );

        let main_sample_count = if matches!(self.render_mode, RenderMode::RasterMultisampled) {
            vk::SampleCountFlags::N4
        } else {
            vk::SampleCountFlags::N1
        };
        let depth_image_desc = ImageDesc::new_2d(
            swap_extent.width,
            swap_extent.height,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        )
        .with_samples(main_sample_count);
        let depth_image = schedule.describe_image(&depth_image_desc);
        let mut main_render_state =
            RenderState::new(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]).with_depth_temp(depth_image);
        if main_sample_count != vk::SampleCountFlags::N1 {
            let msaa_image = schedule.describe_image(
                &ImageDesc::new_2d(
                    swap_extent.width,
                    swap_extent.height,
                    swap_format,
                    vk::ImageAspectFlags::COLOR,
                )
                .with_samples(main_sample_count),
            );
            main_render_state = main_render_state.with_color_temp(msaa_image);
        }

        let mesh_info = *self.mesh_info.lock().unwrap();
        let position_buffer = base.systems.resource_loader.get_buffer(mesh_info.position_buffer);
        let attribute_buffer = base.systems.resource_loader.get_buffer(mesh_info.attribute_buffer);
        let index_buffer = base.systems.resource_loader.get_buffer(mesh_info.index_buffer);
        let instance_buffer = base.systems.resource_loader.get_buffer(mesh_info.instance_buffer);

        let view_from_world = Isometry3::new(
            Vec3::new(0.0, 0.0, -6.0),
            Rotor3::from_rotation_yz(0.5) * Rotor3::from_rotation_xz(self.angle),
        );

        let vertical_fov = PI / 7.0;
        let aspect_ratio = (swap_extent.width as f32) / (swap_extent.height as f32);
        let proj_from_view = projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);

        if base.context.device.extensions.khr_acceleration_structure
            && self.accel_info.is_none()
            && position_buffer.is_some()
            && attribute_buffer.is_some()
            && index_buffer.is_some()
        {
            self.accel_info = Some(AccelInfo::new(
                &base.context,
                &base.systems.pipeline_cache,
                &base.systems.descriptor_pool,
                &mut base.systems.resource_loader,
                &mesh_info,
                &mut base.systems.global_allocator,
                &mut schedule,
            ));
        }
        if let Some(accel_info) = self.accel_info.as_mut() {
            accel_info.update(
                &base.context,
                &base.systems.resource_loader,
                &mut base.systems.global_allocator,
                &mut schedule,
            );
        }
        let trace_image = if let Some(accel_info) = self
            .accel_info
            .as_ref()
            .filter(|_| matches!(self.render_mode, RenderMode::RayTrace))
        {
            let world_from_view = view_from_world.inversed();

            let xy_from_st =
                Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * vertical_fov).tan(), Vec2::zero());
            let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, 2.0), Vec2::new(1.0, -1.0));
            let coord_from_uv = Scale2Offset2::new(
                UVec2::new(swap_extent.width, swap_extent.height).as_float(),
                Vec2::broadcast(-0.5), // coord is pixel centre
            );
            let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

            let ray_origin = world_from_view.translation;
            let ray_vec_from_coord = world_from_view.rotation.into_matrix()
                * Mat3::from_scale(-1.0)
                * xy_from_coord.into_homogeneous_matrix();

            accel_info.dispatch(
                &base.context,
                &base.systems.resource_loader,
                &mut schedule,
                &base.systems.descriptor_pool,
                &swap_extent,
                ray_origin,
                ray_vec_from_coord,
            )
        } else {
            None
        };

        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |params| {
                if let Some(trace_image) = trace_image {
                    params.add_image(trace_image, ImageUsage::FRAGMENT_STORAGE_READ);
                }
            },
            {
                let context = &base.context;
                let descriptor_pool = &base.systems.descriptor_pool;
                let pipeline_cache = &base.systems.pipeline_cache;
                let copy_descriptor_set_layout = &self.copy_descriptor_set_layout;
                let copy_pipeline_layout = self.copy_pipeline_layout;
                let test_descriptor_set_layout = &self.test_descriptor_set_layout;
                let test_pipeline_layout = self.test_pipeline_layout;
                let window = &base.window;
                let ui_platform = &mut base.ui_platform;
                let ui_renderer = &mut base.ui_renderer;
                move |params, cmd, render_pass| {
                    set_viewport_helper(&context.device, cmd, swap_extent);

                    if let Some(trace_image) = trace_image {
                        let trace_image_view = params.get_image_view(trace_image);

                        let copy_descriptor_set = copy_descriptor_set_layout.write(&descriptor_pool, trace_image_view);

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count);

                        draw_helper(
                            &context.device,
                            pipeline_cache,
                            cmd,
                            copy_pipeline_layout,
                            &state,
                            "mesh/copy.vert.spv",
                            "mesh/copy.frag.spv",
                            copy_descriptor_set,
                            3,
                        );
                    } else if let (
                        Some(position_buffer),
                        Some(attribute_buffer),
                        Some(index_buffer),
                        Some(instance_buffer),
                    ) = (position_buffer, attribute_buffer, index_buffer, instance_buffer)
                    {
                        let test_descriptor_set = test_descriptor_set_layout.write(&descriptor_pool, &|buf| {
                            *buf = TestData {
                                proj_from_world: *(proj_from_view * view_from_world.into_homogeneous_matrix())
                                    .as_array(),
                            };
                        });

                        let state = GraphicsPipelineState::new(render_pass, main_sample_count).with_vertex_inputs(
                            &[
                                vk::VertexInputBindingDescription {
                                    binding: 0,
                                    stride: mem::size_of::<PositionData>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                vk::VertexInputBindingDescription {
                                    binding: 1,
                                    stride: mem::size_of::<AttributeData>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                vk::VertexInputBindingDescription {
                                    binding: 2,
                                    stride: mem::size_of::<InstanceData>() as u32,
                                    input_rate: vk::VertexInputRate::INSTANCE,
                                },
                            ],
                            &[
                                vk::VertexInputAttributeDescription {
                                    location: 0,
                                    binding: 0,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 1,
                                    binding: 1,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 2,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 0,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 3,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 16,
                                },
                                vk::VertexInputAttributeDescription {
                                    location: 4,
                                    binding: 2,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    offset: 32,
                                },
                            ],
                        );
                        let pipeline = pipeline_cache.get_graphics(
                            "mesh/raster.vert.spv",
                            "mesh/raster.frag.spv",
                            test_pipeline_layout,
                            &state,
                        );
                        unsafe {
                            context
                                .device
                                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                            context.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                test_pipeline_layout,
                                0,
                                slice::from_ref(&test_descriptor_set),
                                &[],
                            );
                            context.device.cmd_bind_vertex_buffers(
                                cmd,
                                0,
                                &[position_buffer, attribute_buffer, instance_buffer],
                                &[0, 0, 0],
                            );
                            context
                                .device
                                .cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
                            context.device.cmd_draw_indexed(
                                cmd,
                                mesh_info.triangle_count * 3,
                                MeshInfo::INSTANCE_COUNT as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }

                    // draw imgui
                    ui_platform.prepare_render(&ui, window);

                    let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                    ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
                }
            },
        );

        schedule.run(
            &base.context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            swap_image,
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display.present(swap_vk_image, rendering_finished_semaphore);

        if self.is_rotating {
            self.angle += base.ui_context.io().delta_time;
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let device = self.context.device;
        unsafe {
            device.destroy_pipeline_layout(Some(self.test_pipeline_layout), None);
            device.destroy_descriptor_set_layout(Some(self.test_descriptor_set_layout.0), None);

            device.destroy_pipeline_layout(Some(self.copy_pipeline_layout), None);
            device.destroy_descriptor_set_layout(Some(self.copy_descriptor_set_layout.0), None);

            if let Some(accel_info) = self.accel_info.take() {
                if let Some(top_level) = accel_info.top_level {
                    device.destroy_acceleration_structure_khr(Some(top_level.accel), None);
                }
                device.destroy_acceleration_structure_khr(Some(accel_info.bottom_level.accel), None);

                device.destroy_pipeline_layout(Some(accel_info.trace_pipeline_layout), None);
                device.destroy_descriptor_set_layout(Some(accel_info.trace_descriptor_set_layout.0), None);
            }
        }
    }
}

fn main() {
    let mut params = ContextParams {
        version: vk::Version::from_raw_parts(1, 1, 0), // Vulkan 1.1 needed for ray tracing
        allow_ray_tracing: true,
        ..Default::default()
    };
    let mut mesh_file_name = None;
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        match arg {
            "--no-rays" => params.allow_ray_tracing = false,
            _ => {
                if !params.parse_arg(arg) {
                    if mesh_file_name.is_none() {
                        mesh_file_name = Some(arg.to_owned());
                    } else {
                        panic!("unknown argument {:?}", arg);
                    }
                }
            }
        }
    }
    let mesh_file_name = mesh_file_name.expect("missing PLY mesh filename argument");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("mesh")
        .with_inner_size(Size::Logical(LogicalSize::new(1920.0, 1080.0)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &params);
    let app = App::new(&mut base, mesh_file_name);

    let mut apps = Some((base, app));
    event_loop.run(move |event, target, control_flow| {
        match apps
            .as_mut()
            .map(|(base, _)| base)
            .unwrap()
            .process_event(&event, target, control_flow)
        {
            AppEventResult::None => {}
            AppEventResult::Redraw => {
                let (base, app) = apps.as_mut().unwrap();
                app.render(base);
            }
            AppEventResult::Destroy => {
                apps.take();
            }
        }
    });
}
