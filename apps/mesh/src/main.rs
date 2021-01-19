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
    min: Vec3,
    max: Vec3,
    index_count: u32,
}

impl Default for MeshInfo {
    fn default() -> Self {
        Self {
            min: Vec3::broadcast(f32::MAX),
            max: Vec3::broadcast(-f32::MAX),
            index_count: 0,
        }
    }
}

struct App {
    context: Arc<Context>,

    test_descriptor_set_layout: TestDescriptorSetLayout,
    test_pipeline_layout: vk::PipelineLayout,

    vertex_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
    mesh_info: Arc<Mutex<MeshInfo>>,
    use_msaa: bool,
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

        let vertex_buffer = resource_loader.create_buffer();
        let index_buffer = resource_loader.create_buffer();
        let mesh_info = Arc::new(Mutex::new(MeshInfo::default()));
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

                let vertex_buffer_desc = BufferDesc::new((vertices.len() * mem::size_of::<PlyVertex>()) as u32);
                let mut mapping = allocator
                    .map_buffer::<PlyVertex>(vertex_buffer, &vertex_buffer_desc, BufferUsage::VERTEX_BUFFER)
                    .unwrap();
                let mut mi = MeshInfo::default();
                for (dst, src) in mapping.get_mut().iter_mut().zip(vertices.iter()) {
                    *dst = *src;
                    let v = Vec3::new(src.x, src.y, src.z);
                    mi.min = mi.min.min_by_component(v);
                    mi.max = mi.max.max_by_component(v);
                }
                mi.index_count = (3 * faces.len()) as u32;
                *mesh_info.lock().unwrap() = mi;

                let index_buffer_desc = BufferDesc::new((faces.len() * mem::size_of::<PlyFace>()) as u32);
                let mut mapping = allocator
                    .map_buffer::<PlyFace>(index_buffer, &index_buffer_desc, BufferUsage::INDEX_BUFFER)
                    .unwrap();
                mapping.get_mut().copy_from_slice(&faces);
            }
        });

        Self {
            context: Arc::clone(&context),
            test_descriptor_set_layout,
            test_pipeline_layout,
            vertex_buffer,
            index_buffer,
            mesh_info,
            use_msaa: false,
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
        let resource_loader = &base.systems.resource_loader;

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

        let vertex_buffer = resource_loader.get_buffer(self.vertex_buffer);
        let index_buffer = resource_loader.get_buffer(self.index_buffer);
        let mesh_info = self.mesh_info.lock().unwrap().clone();
        schedule.add_graphics(
            command_name!("main"),
            main_render_state,
            |_params| {},
            |_params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_extent);

                if let (Some(vertex_buffer), Some(index_buffer)) = (vertex_buffer, index_buffer) {
                    let centre = 0.5 * (mesh_info.max + mesh_info.min);
                    let half_size = 0.5 * (mesh_info.max - mesh_info.min).component_max();
                    let view_from_local = Mat4::look_at(
                        centre + half_size * Vec3::new(6.0 * self.angle.cos(), 3.0, 6.0 * self.angle.sin()),
                        centre,
                        Vec3::unit_y(),
                    );

                    let aspect_ratio = (swap_extent.width as f32) / (swap_extent.height as f32);
                    let proj_from_view =
                        uv::projection::rh_yup::perspective_reversed_infinite_z_vk(0.5, aspect_ratio, 0.1);

                    let test_descriptor_set = self.test_descriptor_set_layout.write(&descriptor_pool, &|buf| {
                        *buf = TestData {
                            proj_from_local: *(proj_from_view * view_from_local).as_array(),
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
                        context.device.cmd_draw_indexed(cmd, mesh_info.index_count, 1, 0, 0, 0);
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
        }
    }
}

fn main() {
    let mut params = ContextParams::default();
    params.enable_geometry_shader = true; // for gl_PrimitiveID in fragment shader
    let mut mesh_file_name = None;
    for arg in env::args().skip(1) {
        let arg = arg.as_str();
        match arg {
            _ => {
                if !params.parse_arg(arg) {
                    if !mesh_file_name.is_some() {
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
