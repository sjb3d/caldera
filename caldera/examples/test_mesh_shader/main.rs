mod cluster;
mod loader;

use crate::{cluster::*, loader::*};
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;
use spark::vk;
use std::{
    mem,
    path::{Path, PathBuf},
    slice,
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
struct PackedTransform {
    translation: Vec3,
    scale: f32,
    rotation_quat: [f32; 4],
}

impl From<Similarity3> for PackedTransform {
    fn from(s: Similarity3) -> Self {
        Self {
            translation: s.translation,
            scale: s.scale,
            rotation_quat: s.rotation.into_quaternion_array(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct StandardUniforms {
    proj_from_view: Mat4,
    view_from_local: PackedTransform,
}

descriptor_set!(StandardDescriptorSet {
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
    proj_from_view: Mat4,
    view_from_local: PackedTransform,
    do_backface_culling: u32,
    task_count: u32,
}

descriptor_set!(ClusterDescriptorSet {
    cluster_uniforms: UniformData<ClusterUniforms>,
    position: StorageBuffer,
    normal: StorageBuffer,
    cluster_desc: StorageBuffer,
});

struct MeshInfo {
    triangle_count: u32,
    cluster_count: u32,
    min: Vec3,
    max: Vec3,
    position_buffer: vk::Buffer,
    normal_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    cluster_buffer: vk::Buffer,
}

impl MeshInfo {
    fn get_world_from_local(&self) -> Similarity3 {
        let scale = 0.9 / (self.max.y - self.min.y);
        let offset = (-0.5 * scale) * (self.max + self.min);
        Similarity3::new(offset, Rotor3::identity(), scale)
    }

    async fn load(resource_loader: ResourceLoader, mesh_file_name: &Path, with_mesh_shader: bool) -> Self {
        let mesh_buffer_usage = if with_mesh_shader {
            BufferUsage::MESH_STORAGE_READ
        } else {
            BufferUsage::empty()
        };

        let mesh = load_ply_mesh(&mesh_file_name);
        println!(
            "loaded mesh: {} vertices, {} triangles",
            mesh.positions.len(),
            mesh.triangles.len()
        );
        let (mesh, clusters) = build_clusters(mesh);

        let position_buffer_desc = BufferDesc::new(mesh.positions.len() * mem::size_of::<Vec3>());
        let mut writer = resource_loader
            .buffer_writer(&position_buffer_desc, BufferUsage::VERTEX_BUFFER | mesh_buffer_usage)
            .await;
        let mut min = Vec3::broadcast(f32::MAX);
        let mut max = Vec3::broadcast(f32::MIN);
        for &pos in &mesh.positions {
            writer.write(&pos);
            min = min.min_by_component(pos);
            max = max.max_by_component(pos);
        }
        let position_buffer_id = writer.finish();

        let normal_buffer_desc = BufferDesc::new(mesh.normals.len() * mem::size_of::<Vec3>());
        let mut writer = resource_loader
            .buffer_writer(&normal_buffer_desc, BufferUsage::VERTEX_BUFFER | mesh_buffer_usage)
            .await;
        for &normal in &mesh.normals {
            writer.write(&normal);
        }
        let normal_buffer_id = writer.finish();

        let index_buffer_desc = BufferDesc::new(mesh.triangles.len() * mem::size_of::<UVec3>());
        let mut writer = resource_loader
            .buffer_writer(&index_buffer_desc, BufferUsage::INDEX_BUFFER)
            .await;
        for &tri in &mesh.triangles {
            writer.write(&tri);
        }
        let index_buffer_id = writer.finish();

        let cluster_buffer_desc = BufferDesc::new(clusters.len() * mem::size_of::<ClusterDesc>());
        let mut writer = resource_loader
            .buffer_writer(&cluster_buffer_desc, mesh_buffer_usage)
            .await;
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
        let cluster_buffer_id = writer.finish();

        Self {
            triangle_count: mesh.triangles.len() as u32,
            cluster_count: clusters.len() as u32,
            min,
            max,
            position_buffer: resource_loader.get_buffer(position_buffer_id.await),
            normal_buffer: resource_loader.get_buffer(normal_buffer_id.await),
            index_buffer: resource_loader.get_buffer(index_buffer_id.await),
            cluster_buffer: resource_loader.get_buffer(cluster_buffer_id.await),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RenderMode {
    Standard,
    Clusters,
}

struct App {
    has_mesh_shader: bool,
    task_group_size: u32,
    mesh_info: TaskOutput<MeshInfo>,
    render_mode: RenderMode,
    do_backface_culling: bool,
    is_rotating: bool,
    angle: f32,
}

impl App {
    fn new(base: &mut AppBase, mesh_file_name: PathBuf) -> Self {
        let has_mesh_shader = base.context.physical_device_features.mesh_shader.mesh_shader.as_bool()
            && base
                .context
                .physical_device_features
                .subgroup_size_control
                .subgroup_size_control
                .as_bool();
        let task_group_size = base
            .context
            .physical_device_extra_properties
            .as_ref()
            .unwrap()
            .subgroup_size_control
            .max_subgroup_size;
        println!("task group size: {}", task_group_size);

        let resource_loader = base.systems.resource_loader.clone();
        let mesh_info = base
            .systems
            .task_system
            .spawn_task(async move { MeshInfo::load(resource_loader, &mesh_file_name, has_mesh_shader).await });

        Self {
            has_mesh_shader,
            task_group_size,
            mesh_info,
            render_mode: if has_mesh_shader {
                RenderMode::Clusters
            } else {
                RenderMode::Standard
            },
            do_backface_culling: true,
            is_rotating: false,
            angle: 0.0,
        }
    }

    fn render(&mut self, base: &mut AppBase) {
        let cbar = base.systems.acquire_command_buffer();

        base.ui_begin_frame();
        base.egui_ctx.clone().input(|i| {
            if i.key_pressed(egui::Key::Escape) {
                base.exit_requested = true;
            }
        });
        egui::Window::new("Debug")
            .default_pos([5.0, 5.0])
            .default_size([350.0, 150.0])
            .show(&base.egui_ctx, |ui| {
                ui.checkbox(&mut self.is_rotating, "Rotate");
                ui.label("Render Mode:");
                ui.radio_value(&mut self.render_mode, RenderMode::Standard, "Standard");
                if self.has_mesh_shader {
                    ui.radio_value(&mut self.render_mode, RenderMode::Clusters, "Clusters");
                } else {
                    ui.label("Mesh Shaders Not Supported!");
                }
                ui.label("Cluster Settings:");
                ui.checkbox(&mut self.do_backface_culling, "Backface Culling");
            });
        base.systems.draw_ui(&base.egui_ctx);
        base.ui_end_frame(cbar.pre_swapchain_cmd);

        let mut schedule = base.systems.resource_loader.begin_schedule(
            &mut base.systems.render_graph,
            base.context.as_ref(),
            &base.systems.descriptor_pool,
            &base.systems.pipeline_cache,
        );

        let swap_vk_image = base
            .display
            .acquire(&base.window, cbar.image_available_semaphore.unwrap());
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
        let main_render_state = RenderState::new()
            .with_color(swap_image, &[0.1f32, 0.1f32, 0.1f32, 0f32])
            .with_depth(depth_image, AttachmentLoadOp::Clear, AttachmentStoreOp::None);

        let view_from_world = Similarity3::new(
            Vec3::new(0.0, 0.0, -2.5),
            Rotor3::from_rotation_yz(0.2) * Rotor3::from_rotation_xz(self.angle),
            1.0,
        );
        let vertical_fov = PI / 7.0;
        let aspect_ratio = (swap_size.x as f32) / (swap_size.y as f32);
        let proj_from_view = projection::rh_yup::perspective_reversed_infinite_z_vk(vertical_fov, aspect_ratio, 0.1);

        let mesh_info = self.mesh_info.get();

        schedule.add_graphics(command_name!("main"), main_render_state, |_params| {}, {
            let context = base.context.as_ref();
            let descriptor_pool = &base.systems.descriptor_pool;
            let pipeline_cache = &base.systems.pipeline_cache;
            let task_group_size = self.task_group_size;
            let render_mode = self.render_mode;
            let do_backface_culling = self.do_backface_culling;
            let pixels_per_point = base.egui_ctx.pixels_per_point();
            let egui_renderer = &mut base.egui_renderer;
            move |_params, cmd, render_pass| {
                set_viewport_helper(&context.device, cmd, swap_size);

                if let Some(mesh_info) = mesh_info {
                    let world_from_local = mesh_info.get_world_from_local();
                    let view_from_local = view_from_world * world_from_local;

                    match render_mode {
                        RenderMode::Standard => {
                            let standard_descriptor_set = StandardDescriptorSet::create(descriptor_pool, |buf| {
                                *buf = StandardUniforms {
                                    proj_from_view,
                                    view_from_local: view_from_local.into(),
                                }
                            });

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

                            let standard_pipeline_layout =
                                pipeline_cache.get_pipeline_layout(slice::from_ref(&standard_descriptor_set.layout));
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
                                    slice::from_ref(&standard_descriptor_set.set),
                                    &[],
                                );
                                context.device.cmd_bind_vertex_buffers(
                                    cmd,
                                    0,
                                    &[mesh_info.position_buffer, mesh_info.normal_buffer],
                                    &[0, 0],
                                );
                                context.device.cmd_bind_index_buffer(
                                    cmd,
                                    Some(mesh_info.index_buffer),
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                context
                                    .device
                                    .cmd_draw_indexed(cmd, mesh_info.triangle_count * 3, 1, 0, 0, 0);
                            }
                        }
                        RenderMode::Clusters => {
                            // draw cluster test
                            let task_count = mesh_info.cluster_count;
                            let cluster_descriptor_set = ClusterDescriptorSet::create(
                                descriptor_pool,
                                |buf: &mut ClusterUniforms| {
                                    *buf = ClusterUniforms {
                                        proj_from_view,
                                        view_from_local: view_from_local.into(),
                                        do_backface_culling: if do_backface_culling { 1 } else { 0 },
                                        task_count,
                                    }
                                },
                                mesh_info.position_buffer,
                                mesh_info.normal_buffer,
                                mesh_info.cluster_buffer,
                            );

                            let state = GraphicsPipelineState::new(render_pass, main_sample_count);
                            let cluster_pipeline_layout =
                                pipeline_cache.get_pipeline_layout(slice::from_ref(&cluster_descriptor_set.layout));
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
                                    slice::from_ref(&cluster_descriptor_set.set),
                                    &[],
                                );
                                device.cmd_draw_mesh_tasks_nv(cmd, task_count.div_round_up(task_group_size), 0);
                            }
                        }
                    }
                }

                // draw ui
                let egui_pipeline = pipeline_cache.get_ui(egui_renderer, render_pass, main_sample_count);
                egui_renderer.render(
                    &context.device,
                    cmd,
                    egui_pipeline,
                    swap_size.x,
                    swap_size.y,
                    pixels_per_point,
                );
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

        if self.is_rotating {
            self.angle += base.egui_ctx.input(|i| i.stable_dt);
        }
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

    /// Whether to use NV_mesh_shader
    #[structopt(long, possible_values=&ContextFeature::VARIANTS, default_value="optional")]
    mesh_shader: ContextFeature,

    /// The PLY file to load
    mesh_file_name: PathBuf,
}

fn main() {
    let app_params = AppParams::from_args();
    let context_params = ContextParams {
        version: app_params.version,
        inline_uniform_block: app_params.inline_uniform_block,
        mesh_shader: app_params.mesh_shader,
        subgroup_size_control: app_params.mesh_shader,
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
