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
#[repr(C)]
struct PlyVertex {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PlyFace {
    indices: [u32; 3],
}

#[repr(C)]
struct TestData {
    proj_from_local: [f32; 16],
}

descriptor_set_layout!(TestDescriptorSetLayout {
    test: UniformData<TestData>,
});

#[repr(C)]
struct TraceData {
    ray_origin: [f32; 3],
    ray_vec_from_coord: [f32; 9],
}

descriptor_set_layout!(TraceDescriptorSetLayout {
    trace: UniformData<TraceData>,
    accel: AccelerationStructure,
    output: StorageImage,
});

descriptor_set_layout!(CopyDescriptorSetLayout { ids: StorageImage });

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.x = v,
            ("y", ply::Property::Float(v)) => self.y = v,
            ("z", ply::Property::Float(v)) => self.z = v,
            _ => {}
        }
    }
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        Self { indices: [0, 0, 0] }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(v)) => {
                assert_eq!(v.len(), 3);
                for (dst, src) in self.indices.iter_mut().zip(v.iter().rev()) {
                    *dst = *src as u32;
                }
            }
            (k, _) => panic!("unknown key {}", k),
        }
    }
}

#[derive(Clone, Copy)]
struct MeshInfo {
    vertex_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
    min: Vec3,
    max: Vec3,
    vertex_count: u32,
    triangle_count: u32,
}

impl MeshInfo {
    fn new(resource_loader: &mut ResourceLoader) -> Self {
        Self {
            vertex_buffer: resource_loader.create_buffer(),
            index_buffer: resource_loader.create_buffer(),
            min: Vec3::broadcast(f32::MAX),
            max: Vec3::broadcast(-f32::MAX),
            vertex_count: 0,
            triangle_count: 0,
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
        let vertex_buffer = resource_loader.get_buffer(mesh_info.vertex_buffer).unwrap();
        let index_buffer = resource_loader.get_buffer(mesh_info.index_buffer).unwrap();

        let vertex_buffer_address = {
            let info = vk::BufferDeviceAddressInfo {
                buffer: Some(vertex_buffer),
                ..Default::default()
            };
            unsafe { context.device.get_buffer_device_address(&info) }
        };
        let index_buffer_address = {
            let info = vk::BufferDeviceAddressInfo {
                buffer: Some(index_buffer),
                ..Default::default()
            };
            unsafe { context.device.get_buffer_device_address(&info) }
        };

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

                let scratch_buffer_address = {
                    let info = vk::BufferDeviceAddressInfo {
                        buffer: Some(scratch_buffer),
                        ..Default::default()
                    };
                    unsafe { context.device.get_buffer_device_address(&info) }
                };

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
        let instance_buffer_address = {
            let info = vk::BufferDeviceAddressInfo {
                buffer: Some(instance_buffer),
                ..Default::default()
            };
            unsafe { context.device.get_buffer_device_address(&info) }
        };

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

        let instance_count = 1;
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

                let scratch_buffer_address = {
                    let info = vk::BufferDeviceAddressInfo {
                        buffer: Some(scratch_buffer),
                        ..Default::default()
                    };
                    unsafe { context.device.get_buffer_device_address(&info) }
                };

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

struct AccelInfo {
    trace_descriptor_set_layout: TraceDescriptorSetLayout,
    trace_pipeline_layout: vk::PipelineLayout,
    trace_pipeline: vk::Pipeline,
    shader_binding_table: StaticBufferHandle,
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

        // TODO: figure out live reload, needs to regenerate SBT!
        let trace_pipeline = pipeline_cache.get_ray_tracing(
            "mesh/trace.rgen.spv",
            "mesh/trace.rchit.spv",
            "mesh/trace.rmiss.spv",
            trace_pipeline_layout,
        );

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

                let section_size = rtpp
                    .shader_group_handle_size
                    .max(rtpp.shader_group_handle_alignment)
                    .max(rtpp.shader_group_base_alignment) as usize;

                let desc = BufferDesc::new(3 * section_size as u32);
                let mut mapping = allocator
                    .map_buffer::<u8>(shader_binding_table, &desc, BufferUsage::SHADER_BINDING_TABLE)
                    .unwrap();

                for (dst, src) in mapping
                    .get_mut()
                    .chunks_exact_mut(section_size)
                    .zip(handle_data.chunks_exact(handle_size))
                {
                    dst[..handle_size].copy_from_slice(src);
                }
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
            move |allocator| {
                let desc = BufferDesc::new(mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u32);
                let mut mapping = allocator
                    .map_buffer::<vk::AccelerationStructureInstanceKHR>(
                        instance_buffer,
                        &desc,
                        BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT,
                    )
                    .unwrap();

                mapping
                    .get_mut()
                    .copy_from_slice(&[vk::AccelerationStructureInstanceKHR {
                        transform: vk::TransformMatrixKHR {
                            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        },
                        instance_custom_index_and_mask: 0xff_00_00_00,
                        instance_shader_binding_table_record_offset_and_flags: 0,
                        acceleration_structure_reference: bottom_level_device_address,
                    }]);
            }
        });

        Self {
            trace_descriptor_set_layout,
            trace_pipeline_layout,
            trace_pipeline,
            shader_binding_table,
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

        let shader_binding_table_address = {
            let info = vk::BufferDeviceAddressInfo {
                buffer: Some(shader_binding_table_buffer),
                ..Default::default()
            };
            unsafe { context.device.get_buffer_device_address(&info) }
        };

        let output_desc = ImageDesc::new_2d(
            swap_extent.width,
            swap_extent.height,
            vk::Format::R32_UINT,
            vk::ImageAspectFlags::COLOR,
        );
        let output_image = schedule.describe_image(&output_desc);

        let rtpp = context.ray_tracing_pipeline_properties.as_ref().unwrap();
        let section_size = rtpp
            .shader_group_handle_size
            .max(rtpp.shader_group_handle_alignment)
            .max(rtpp.shader_group_base_alignment) as u64;

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

                let raygen_shader_binding_table = vk::StridedDeviceAddressRegionKHR {
                    device_address: shader_binding_table_address,
                    stride: section_size,
                    size: section_size,
                };
                let miss_shader_binding_table = vk::StridedDeviceAddressRegionKHR {
                    device_address: shader_binding_table_address + section_size,
                    stride: section_size,
                    size: section_size,
                };
                let hit_shader_binding_table = vk::StridedDeviceAddressRegionKHR {
                    device_address: shader_binding_table_address + 2 * section_size,
                    stride: section_size,
                    size: section_size,
                };
                let callable_shader_binding_table = vk::StridedDeviceAddressRegionKHR {
                    device_address: 0,
                    stride: 0,
                    size: 0,
                };
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

struct App {
    context: Arc<Context>,

    test_descriptor_set_layout: TestDescriptorSetLayout,
    test_pipeline_layout: vk::PipelineLayout,

    copy_descriptor_set_layout: CopyDescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,

    mesh_info: Arc<Mutex<MeshInfo>>,
    accel_info: Option<AccelInfo>,
    use_msaa: bool,
    show_trace: bool,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: String) -> Self {
        let context = &base.context;
        let descriptor_pool = &base.systems.descriptor_pool;
        let resource_loader = &mut base.systems.resource_loader;

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

        let extra_buffer_usage = if context.device.extensions.khr_acceleration_structure {
            BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
        } else {
            BufferUsage::empty()
        };

        let mesh_info = Arc::new(Mutex::new(MeshInfo::new(resource_loader)));
        resource_loader.async_load({
            let mesh_info = Arc::clone(&mesh_info);
            move |allocator| {
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

                let mut mi = *mesh_info.lock().unwrap();

                let vertex_buffer_desc = BufferDesc::new((vertices.len() * mem::size_of::<PlyVertex>()) as u32);
                let mut mapping = allocator
                    .map_buffer::<PlyVertex>(
                        mi.vertex_buffer,
                        &vertex_buffer_desc,
                        BufferUsage::VERTEX_BUFFER | extra_buffer_usage,
                    )
                    .unwrap();
                for (dst, src) in mapping.get_mut().iter_mut().zip(vertices.iter()) {
                    *dst = *src;
                    let v = Vec3::new(src.x, src.y, src.z);
                    mi.min = mi.min.min_by_component(v);
                    mi.max = mi.max.max_by_component(v);
                }
                mi.vertex_count = vertices.len() as u32;

                let index_buffer_desc = BufferDesc::new((faces.len() * mem::size_of::<PlyFace>()) as u32);
                let mut mapping = allocator
                    .map_buffer::<PlyFace>(
                        mi.index_buffer,
                        &index_buffer_desc,
                        BufferUsage::INDEX_BUFFER | extra_buffer_usage,
                    )
                    .unwrap();
                mapping.get_mut().copy_from_slice(&faces);
                mi.triangle_count = faces.len() as u32;

                *mesh_info.lock().unwrap() = mi;
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
            use_msaa: false,
            show_trace: true,
            angle: 0.0,
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
            .build(&ui, || {
                ui.checkbox(im_str!("Use MSAA"), &mut self.use_msaa);
                ui.checkbox(im_str!("Show Trace"), &mut self.show_trace);
            });

        let cbar = base.systems.acquire_command_buffer();
        let ui_platform = &mut base.ui_platform;
        let ui_renderer = &mut base.ui_renderer;
        let window = &base.window;
        ui_renderer.begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut ui = Some(ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let context = &base.context;
        let descriptor_pool = &base.systems.descriptor_pool;
        let pipeline_cache = &base.systems.pipeline_cache;
        let resource_loader = &mut base.systems.resource_loader;
        let global_allocator = &mut base.systems.global_allocator;

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

        let main_sample_count = if self.use_msaa {
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
            RenderState::new(swap_image, &[0f32, 0f32, 0f32, 0f32]).with_depth_temp(depth_image);
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
        let vertex_buffer = resource_loader.get_buffer(mesh_info.vertex_buffer);
        let index_buffer = resource_loader.get_buffer(mesh_info.index_buffer);

        let centre = 0.5 * (mesh_info.max + mesh_info.min);
        let half_size = 0.5 * (mesh_info.max - mesh_info.min).component_max();
        let world_from_local = uv::Isometry3::new(-centre, uv::Rotor3::identity());
        let view_from_world = uv::Isometry3::new(
            Vec3::new(0.0, 0.0, -8.0 * half_size),
            uv::Rotor3::from_rotation_xz(self.angle) * uv::Rotor3::from_rotation_xy(0.5),
        );
        let view_from_local = view_from_world * world_from_local;

        let vertical_fov = 0.5;
        let aspect_ratio = (swap_extent.width as f32) / (swap_extent.height as f32);
        let proj_from_view =
            uv::projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);

        if context.device.extensions.khr_acceleration_structure
            && self.accel_info.is_none()
            && vertex_buffer.is_some()
            && index_buffer.is_some()
        {
            self.accel_info = Some(AccelInfo::new(
                &base.context,
                pipeline_cache,
                descriptor_pool,
                resource_loader,
                &mesh_info,
                global_allocator,
                &mut schedule,
            ));
        }
        if let Some(accel_info) = self.accel_info.as_mut() {
            accel_info.update(&base.context, resource_loader, global_allocator, &mut schedule);
        }
        let trace_image = if let Some(accel_info) = self.accel_info.as_ref().filter(|_| self.show_trace) {
            let local_from_view = view_from_local.inversed();

            let xy_from_st =
                Scale2Offset2::new(Vec2::new(aspect_ratio, 1.0) * (0.5 * vertical_fov).tan(), Vec2::zero());
            let st_from_uv = Scale2Offset2::new(Vec2::new(-2.0, 2.0), Vec2::new(1.0, -1.0));
            let coord_from_uv = Scale2Offset2::new(
                UVec2::new(swap_extent.width, swap_extent.height).as_float(),
                Vec2::broadcast(0.5),
            );
            let xy_from_coord = xy_from_st * st_from_uv * coord_from_uv.inversed();

            let ray_origin = local_from_view.translation;
            let ray_vec_from_coord = local_from_view.rotation.into_matrix()
                * Mat3::from_scale(-1.0)
                * xy_from_coord.into_homogeneous_matrix();

            accel_info.dispatch(
                &base.context,
                resource_loader,
                &mut schedule,
                &descriptor_pool,
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
            |params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_extent);

                if let Some(trace_image) = trace_image {
                    let trace_image_view = params.get_image_view(trace_image);

                    let copy_descriptor_set = self
                        .copy_descriptor_set_layout
                        .write(&descriptor_pool, trace_image_view);

                    let state = GraphicsPipelineState::new(render_pass, main_sample_count);

                    draw_helper(
                        &context.device,
                        pipeline_cache,
                        cmd,
                        self.copy_pipeline_layout,
                        &state,
                        "mesh/copy.vert.spv",
                        "mesh/copy.frag.spv",
                        copy_descriptor_set,
                        3,
                    );
                } else if let (Some(vertex_buffer), Some(index_buffer)) = (vertex_buffer, index_buffer) {
                    let test_descriptor_set = self.test_descriptor_set_layout.write(&descriptor_pool, &|buf| {
                        *buf = TestData {
                            proj_from_local: *(proj_from_view * view_from_local.into_homogeneous_matrix()).as_array(),
                        };
                    });

                    let state = GraphicsPipelineState::new(render_pass, main_sample_count).with_vertex_inputs(
                        &[vk::VertexInputBindingDescription {
                            binding: 0,
                            stride: mem::size_of::<PlyVertex>() as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        }],
                        &[vk::VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32B32_SFLOAT,
                            offset: 0,
                        }],
                    );
                    let pipeline = pipeline_cache.get_graphics(
                        "mesh/test.vert.spv",
                        "mesh/test.frag.spv",
                        self.test_pipeline_layout,
                        &state,
                    );
                    unsafe {
                        context
                            .device
                            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                        context.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.test_pipeline_layout,
                            0,
                            slice::from_ref(&test_descriptor_set),
                            &[],
                        );
                        context
                            .device
                            .cmd_bind_vertex_buffers(cmd, 0, slice::from_ref(&vertex_buffer), &[0]);
                        context
                            .device
                            .cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
                        context
                            .device
                            .cmd_draw_indexed(cmd, mesh_info.triangle_count * 3, 1, 0, 0, 0);
                    }
                }

                if let Some(ui) = ui.take() {
                    ui_platform.prepare_render(&ui, window);

                    let pipeline = pipeline_cache.get_ui(&ui_renderer, render_pass, main_sample_count);
                    ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
                }
            },
        );

        schedule.run(
            &context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            swap_image,
            &mut base.systems.query_pool,
        );
        drop(ui);

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display.present(swap_vk_image, rendering_finished_semaphore);

        self.angle += base.ui_context.io().delta_time;
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
        enable_geometry_shader: true, // for gl_PrimitiveID in fragment shader
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
        .with_inner_size(Size::Logical(LogicalSize::new(640.0, 480.0)))
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
