mod cluster;
mod loader;

use crate::{cluster::*, loader::*};
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use imgui::{im_str, Key};
use spark::vk;
use std::{
    mem,
    path::{Path, PathBuf},
    slice,
    sync::{Arc, Mutex},
};
use structopt::StructOpt;
use strum::VariantNames;
use winit::{
    dpi::{LogicalSize, Size},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct StandardUniforms {
    proj_from_local: Mat4,
}

descriptor_set_layout!(StandardDescriptorSetLayout {
    standard_uniforms: UniformData<StandardUniforms>,
});

const MAX_PACKED_INDICES_PER_CLUSTER: usize = (MAX_TRIANGLES_PER_CLUSTER * 3) / 4;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ClusterDesc {
    position_sphere: SphereBounds,
    face_normal_cone: ConeBounds,
    vertex_count: u32,
    triangle_count: u32,
    vertices: [u32; MAX_VERTICES_PER_CLUSTER],
    packed_indices: [u32; MAX_PACKED_INDICES_PER_CLUSTER],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ClusterUniforms {
    proj_from_local: Mat4,
    view_from_local: Mat4,
    view_from_local_rotflip: Mat3,
    view_from_local_scale: f32,
    task_count: u32,
}

descriptor_set_layout!(ClusterDescriptorSetLayout {
    cluster_uniforms: UniformData<ClusterUniforms>,
    position: StorageBuffer,
    normal: StorageBuffer,
    cluster_desc: StorageBuffer,
});

#[derive(Clone, Copy)]
struct MeshInfo {
    triangle_count: u32,
    cluster_count: u32,
    min: Vec3,
    max: Vec3,
    position_buffer: StaticBufferHandle,
    normal_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
    cluster_buffer: StaticBufferHandle,
}

#[derive(Clone, Copy)]
struct MeshBuffers {
    position: vk::Buffer,
    normal: vk::Buffer,
    index: vk::Buffer,
    cluster: vk::Buffer,
}

impl MeshInfo {
    fn new(resource_loader: &mut ResourceLoader) -> Self {
        Self {
            triangle_count: 0,
            cluster_count: 0,
            min: Vec3::zero(),
            max: Vec3::one(),
            position_buffer: resource_loader.create_buffer(),
            normal_buffer: resource_loader.create_buffer(),
            index_buffer: resource_loader.create_buffer(),
            cluster_buffer: resource_loader.create_buffer(),
        }
    }

    fn get_world_from_local(&self) -> Similarity3 {
        let scale = 0.9 / (self.max - self.min).component_max();
        let offset = (-0.5 * scale) * (self.max + self.min);
        Similarity3::new(offset, Rotor3::identity(), scale)
    }

    fn get_buffers(&self, resource_loader: &ResourceLoader) -> Option<MeshBuffers> {
        if self.triangle_count == 0 {
            return None;
        }
        let position = resource_loader.get_buffer(self.position_buffer)?;
        let normal = resource_loader.get_buffer(self.normal_buffer)?;
        let index = resource_loader.get_buffer(self.index_buffer)?;
        let cluster = resource_loader.get_buffer(self.cluster_buffer)?;
        Some(MeshBuffers {
            position,
            normal,
            index,
            cluster,
        })
    }

    fn load(&mut self, allocator: &mut ResourceAllocator, mesh_file_name: &Path) {
        let mesh = load_ply_mesh(&mesh_file_name);
        println!(
            "loaded mesh: {} vertices, {} triangles",
            mesh.positions.len(),
            mesh.triangles.len()
        );
        let (mesh, clusters) = build_clusters(mesh);

        let position_buffer_desc = BufferDesc::new(mesh.positions.len() * mem::size_of::<Vec3>());
        let mut writer = allocator
            .map_buffer(
                self.position_buffer,
                &position_buffer_desc,
                BufferUsage::VERTEX_BUFFER | BufferUsage::MESH_STORAGE_READ,
            )
            .unwrap();
        self.min = Vec3::broadcast(f32::MAX);
        self.max = Vec3::broadcast(f32::MIN);
        for &pos in &mesh.positions {
            writer.write(&pos);
            self.min = self.min.min_by_component(pos);
            self.max = self.max.max_by_component(pos);
        }

        let normal_buffer_desc = BufferDesc::new(mesh.normals.len() * mem::size_of::<Vec3>());
        let mut writer = allocator
            .map_buffer(
                self.normal_buffer,
                &normal_buffer_desc,
                BufferUsage::VERTEX_BUFFER | BufferUsage::MESH_STORAGE_READ,
            )
            .unwrap();
        for &normal in &mesh.normals {
            writer.write(&normal);
        }

        let index_buffer_desc = BufferDesc::new(mesh.triangles.len() * mem::size_of::<UVec3>());
        let mut writer = allocator
            .map_buffer(self.index_buffer, &index_buffer_desc, BufferUsage::INDEX_BUFFER)
            .unwrap();
        for &tri in &mesh.triangles {
            writer.write(&tri);
        }
        self.triangle_count = mesh.triangles.len() as u32;

        let cluster_buffer_desc = BufferDesc::new(clusters.len() * mem::size_of::<ClusterDesc>());
        let mut writer = allocator
            .map_buffer(
                self.cluster_buffer,
                &cluster_buffer_desc,
                BufferUsage::MESH_STORAGE_READ,
            )
            .unwrap();
        for cluster in &clusters {
            let mut desc = ClusterDesc {
                position_sphere: SphereBounds::from_box(cluster.position_bounds),
                face_normal_cone: ConeBounds::from_box(cluster.face_normal_bounds),
                vertex_count: cluster.mesh_vertices.len() as u32,
                triangle_count: cluster.triangles.len() as u32,
                vertices: [0u32; MAX_VERTICES_PER_CLUSTER],
                packed_indices: [0u32; MAX_PACKED_INDICES_PER_CLUSTER],
            };
            for (src, dst) in cluster.mesh_vertices.iter().zip(desc.vertices.iter_mut()) {
                *dst = *src;
            }
            let indices: &mut [u8] = bytemuck::cast_slice_mut(&mut desc.packed_indices);
            for (src, dst) in cluster
                .triangles
                .iter()
                .flat_map(|tri| tri.as_slice().iter())
                .zip(indices.iter_mut())
            {
                *dst = *src as u8;
            }
            writer.write(&desc);
        }
        self.cluster_count = clusters.len() as u32;
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RenderMode {
    Standard,
    Clusters,
}

struct App {
    context: SharedContext,

    standard_descriptor_set_layout: StandardDescriptorSetLayout,
    standard_pipeline_layout: vk::PipelineLayout,
    cluster_descriptor_set_layout: ClusterDescriptorSetLayout,
    cluster_pipeline_layout: vk::PipelineLayout,

    task_group_size: u32,
    mesh_info: Arc<Mutex<MeshInfo>>,
    render_mode: RenderMode,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: PathBuf) -> Self {
        let context = SharedContext::clone(&base.context);
        let descriptor_set_layout_cache = &mut base.systems.descriptor_set_layout_cache;

        let standard_descriptor_set_layout = StandardDescriptorSetLayout::new(descriptor_set_layout_cache);
        let standard_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(standard_descriptor_set_layout.0);

        let cluster_descriptor_set_layout = ClusterDescriptorSetLayout::new(descriptor_set_layout_cache);
        let cluster_pipeline_layout =
            descriptor_set_layout_cache.create_pipeline_layout(cluster_descriptor_set_layout.0);

        let task_group_size = context
            .physical_device_extra_properties
            .as_ref()
            .unwrap()
            .subgroup_size_control
            .max_subgroup_size;
        println!("task group size: {}", task_group_size);

        let mesh_info = Arc::new(Mutex::new(MeshInfo::new(&mut base.systems.resource_loader)));
        base.systems.resource_loader.async_load({
            let mesh_info = Arc::clone(&mesh_info);
            move |allocator| {
                let mut mesh_info_clone = *mesh_info.lock().unwrap();
                mesh_info_clone.load(allocator, &mesh_file_name);
                *mesh_info.lock().unwrap() = mesh_info_clone;
            }
        });

        Self {
            context,
            standard_descriptor_set_layout,
            standard_pipeline_layout,
            cluster_descriptor_set_layout,
            cluster_pipeline_layout,
            task_group_size,
            mesh_info,
            render_mode: RenderMode::Clusters,
            angle: 0.0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let ui = base.ui_context.frame();
        if ui.is_key_pressed(Key::Escape) {
            base.exit_requested = true;
        }
        imgui::Window::new(im_str!("Debug"))
            .position([5.0, 5.0], imgui::Condition::FirstUseEver)
            .size([350.0, 150.0], imgui::Condition::FirstUseEver)
            .build(&ui, {
                let ui = &ui;
                let render_mode = &mut self.render_mode;
                move || {
                    ui.text("Render Mode:");
                    ui.radio_button(im_str!("Standard"), render_mode, RenderMode::Standard);
                    ui.radio_button(im_str!("Clusters"), render_mode, RenderMode::Clusters);
                }
            });

        let cbar = base.systems.acquire_command_buffer();
        base.ui_renderer
            .begin_frame(&self.context.device, cbar.pre_swapchain_cmd);

        base.systems.draw_ui(&ui);

        let mut schedule = RenderSchedule::new(&mut base.systems.render_graph);

        let swap_vk_image = base.display.acquire(cbar.image_available_semaphore.unwrap());
        let swap_size = base.display.swapchain.get_size();
        let swap_format = base.display.swapchain.get_format();
        let swap_image = schedule.import_image(
            &ImageDesc::new_2d(swap_size, swap_format, vk::ImageAspectFlags::COLOR),
            ImageUsage::COLOR_ATTACHMENT_WRITE | ImageUsage::SWAPCHAIN,
            swap_vk_image,
            ImageUsage::empty(),
            ImageUsage::SWAPCHAIN,
        );

        let main_sample_count = vk::SampleCountFlags::N1;
        let depth_image_desc = ImageDesc::new_2d(swap_size, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH);
        let depth_image = schedule.describe_image(&depth_image_desc);
        let main_render_state =
            RenderState::new(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32]).with_depth_temp(depth_image);

        let mesh_info = *self.mesh_info.lock().unwrap();
        let mesh_buffers = mesh_info.get_buffers(&base.systems.resource_loader);

        let world_from_local = mesh_info.get_world_from_local();
        let view_from_world = Similarity3::new(
            Vec3::new(0.0, 0.0, -3.0),
            Rotor3::from_rotation_yz(0.5) * Rotor3::from_rotation_xz(self.angle),
            1.0,
        );
        let vertical_fov = PI / 7.0;
        let aspect_ratio = (swap_size.x as f32) / (swap_size.y as f32);
        let proj_from_view = projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);
        let view_from_local = view_from_world * world_from_local;
        let proj_from_local = proj_from_view * view_from_local.into_homogeneous_matrix();

        schedule.add_graphics(command_name!("main"), main_render_state, |_params| {}, {
            let context = base.context.as_ref();
            let descriptor_pool = &mut base.systems.descriptor_pool;
            let pipeline_cache = &base.systems.pipeline_cache;
            let task_group_size = self.task_group_size;
            let standard_descriptor_set_layout = &self.standard_descriptor_set_layout;
            let standard_pipeline_layout = self.standard_pipeline_layout;
            let cluster_descriptor_set_layout = &self.cluster_descriptor_set_layout;
            let cluster_pipeline_layout = self.cluster_pipeline_layout;
            let window = &base.window;
            let render_mode = self.render_mode;
            let ui_platform = &mut base.ui_platform;
            let ui_renderer = &mut base.ui_renderer;
            move |_params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_size);

                if let Some(mesh_buffers) = mesh_buffers {
                    match render_mode {
                        RenderMode::Standard => {
                            let standard_descriptor_set = standard_descriptor_set_layout
                                .write(descriptor_pool, |buf| *buf = StandardUniforms { proj_from_local });

                            let state = GraphicsPipelineState::new(render_pass, main_sample_count).with_vertex_inputs(
                                &[
                                    vk::VertexInputBindingDescription {
                                        binding: 0,
                                        stride: mem::size_of::<Vec3>() as u32,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                    vk::VertexInputBindingDescription {
                                        binding: 1,
                                        stride: mem::size_of::<Vec3>() as u32,
                                        input_rate: vk::VertexInputRate::VERTEX,
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
                                ],
                            );

                            let pipeline = pipeline_cache.get_graphics(
                                VertexShaderDesc::standard("test_mesh_shader/standard.vert.spv"),
                                "test_mesh_shader/test.frag.spv",
                                standard_pipeline_layout,
                                &state,
                            );
                            unsafe {
                                context
                                    .device
                                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                                context.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    standard_pipeline_layout,
                                    0,
                                    slice::from_ref(&standard_descriptor_set),
                                    &[],
                                );
                                context.device.cmd_bind_vertex_buffers(
                                    cmd,
                                    0,
                                    &[mesh_buffers.position, mesh_buffers.normal],
                                    &[0, 0],
                                );
                                context
                                    .device
                                    .cmd_bind_index_buffer(cmd, mesh_buffers.index, 0, vk::IndexType::UINT32);
                                context
                                    .device
                                    .cmd_draw_indexed(cmd, mesh_info.triangle_count * 3, 1, 0, 0, 0);
                            }
                        }
                        RenderMode::Clusters => {
                            // draw cluster test
                            let task_count = mesh_info.cluster_count;
                            let cluster_descriptor_set = cluster_descriptor_set_layout.write(
                                descriptor_pool,
                                |buf: &mut ClusterUniforms| {
                                    *buf = ClusterUniforms {
                                        proj_from_local,
                                        view_from_local: view_from_local.into_homogeneous_matrix(),
                                        view_from_local_rotflip: view_from_local.rotation.into_matrix()
                                            * Mat3::from_scale_homogeneous(1.0f32.copysign(view_from_local.scale)),
                                        view_from_local_scale: view_from_local.scale,
                                        task_count,
                                    }
                                },
                                mesh_buffers.position,
                                mesh_buffers.normal,
                                mesh_buffers.cluster,
                            );

                            let state = GraphicsPipelineState::new(render_pass, main_sample_count);
                            let pipeline = pipeline_cache.get_graphics(
                                VertexShaderDesc::mesh(
                                    "test_mesh_shader/cluster.task.spv",
                                    &[SpecializationConstant::new(0, task_group_size)],
                                    Some(task_group_size),
                                    "test_mesh_shader/cluster.mesh.spv",
                                    &[SpecializationConstant::new(0, task_group_size)],
                                ),
                                "test_mesh_shader/test.frag.spv",
                                cluster_pipeline_layout,
                                &state,
                            );
                            let device = &context.device;
                            unsafe {
                                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                                device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    cluster_pipeline_layout,
                                    0,
                                    &[cluster_descriptor_set],
                                    &[],
                                );
                                device.cmd_draw_mesh_tasks_nv(cmd, task_count.div_round_up(task_group_size), 0);
                            }
                        }
                    }
                }

                // draw imgui
                ui_platform.prepare_render(&ui, window);
                let pipeline = pipeline_cache.get_ui(ui_renderer, render_pass, main_sample_count);
                ui_renderer.render(ui.render(), &context.device, cmd, pipeline);
            }
        });

        schedule.run(
            &base.context,
            cbar.pre_swapchain_cmd,
            cbar.post_swapchain_cmd,
            Some(swap_image),
            &mut base.systems.query_pool,
        );

        let rendering_finished_semaphore = base.systems.submit_command_buffer(&cbar);
        base.display
            .present(swap_vk_image, rendering_finished_semaphore.unwrap());

        self.angle += base.ui_context.io().delta_time;
    }
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    /// Core Vulkan version to load
    #[structopt(short, long, parse(try_from_str=try_version_from_str), default_value="1.1")]
    version: vk::Version,

    /// Whether to use EXT_inline_uniform_block
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional")]
    inline_uniform_block: ContextFeature,

    /// The PLY file to load
    mesh_file_name: PathBuf,
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        mesh_shader: ContextFeature::Require,
        subgroup_size_control: ContextFeature::Require,
        ..Default::default()
    };

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("test_mesh_shader")
        .with_inner_size(Size::Logical(LogicalSize::new(1920.0, 1080.0)))
        .build(&event_loop)
        .unwrap();

    let mut base = AppBase::new(window, &context_params);
    let app = App::new(&mut base, app_params.mesh_file_name);

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
