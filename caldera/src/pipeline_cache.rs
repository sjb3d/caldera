use crate::context::*;
use arrayvec::ArrayVec;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use spark::{vk, Builder, Device};
use std::{
    cell::RefCell,
    collections::HashMap,
    convert::TryInto,
    ffi::CStr,
    fs::File,
    io::{self, prelude::*},
    mem,
    path::{Path, PathBuf},
    ptr, slice,
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};

fn read_file_words(path: &Path) -> io::Result<Vec<u32>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes
        .chunks(4)
        .map(|c| u32::from(c[3]) << 24 | u32::from(c[2]) << 16 | u32::from(c[1]) << 8 | u32::from(c[0]))
        .collect())
}

fn load_shader_module(device: &Device, path: &Path) -> Option<vk::ShaderModule> {
    let words = read_file_words(path).ok()?;
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        code_size: words.len() * mem::size_of::<u32>(),
        p_code: words.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_shader_module(&shader_module_create_info, None) }.ok()
}

struct ShaderReloader {
    watcher: RecommendedWatcher,
    join_handle: JoinHandle<()>,
}

struct ShaderLoader {
    context: SharedContext,
    base_path: PathBuf,
    reloader: Option<ShaderReloader>,
    current_shaders: HashMap<PathBuf, vk::ShaderModule>,
    new_shaders: Arc<Mutex<HashMap<PathBuf, vk::ShaderModule>>>,
}

impl ShaderLoader {
    pub fn new<P: AsRef<Path>>(context: &SharedContext, base_path: P) -> Self {
        let base_path = base_path.as_ref().to_owned();

        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::watcher(tx, Duration::from_millis(500)).unwrap();
        watcher.watch(&base_path, RecursiveMode::Recursive).unwrap();

        let current_shaders = HashMap::new();
        let new_shaders = Arc::new(Mutex::new(HashMap::new()));

        let join_handle = thread::spawn({
            let context = SharedContext::clone(context);
            let new_shaders = Arc::clone(&new_shaders);
            let short_base_path = base_path.clone();
            let full_base_path = base_path.canonicalize().unwrap();
            move || {
                while let Ok(event) = rx.recv() {
                    if let DebouncedEvent::Create(path_buf)
                    | DebouncedEvent::Write(path_buf)
                    | DebouncedEvent::Rename(_, path_buf) = event
                    {
                        if let Ok(relative_path) = path_buf.canonicalize().unwrap().strip_prefix(&full_base_path) {
                            if let Some(shader) =
                                load_shader_module(&context.device, &short_base_path.join(relative_path))
                            {
                                let mut new_shaders = new_shaders.lock().unwrap();
                                println!("reloaded shader: {:?}", relative_path);
                                new_shaders.insert(relative_path.to_owned(), shader);
                            }
                        }
                    }
                }
                println!("shader reload stopping!");
            }
        });

        Self {
            context: SharedContext::clone(context),
            base_path,
            reloader: Some(ShaderReloader { watcher, join_handle }),
            current_shaders,
            new_shaders,
        }
    }

    pub fn get_shader<P: AsRef<Path>>(&self, relative_path: P) -> Option<vk::ShaderModule> {
        let relative_path = relative_path.as_ref();
        self.current_shaders.get(relative_path).copied().or_else(|| {
            let mut new_shaders = self.new_shaders.lock().unwrap();
            new_shaders.get(relative_path).copied().or_else(|| {
                load_shader_module(&self.context.device, &self.base_path.join(relative_path)).map(|shader| {
                    new_shaders.insert(relative_path.to_owned(), shader);
                    shader
                })
            })
        })
    }

    pub fn transfer_new_shaders(&mut self) {
        let mut new_shaders = self.new_shaders.lock().unwrap();
        for (k, v) in new_shaders.drain() {
            self.current_shaders.insert(k, v);
        }
    }

    pub fn ui_stats_table_rows(&self, ui: &mut egui::Ui) {
        ui.label("shader");
        ui.label(format!("{}", self.current_shaders.len()));
        ui.end_row();
    }
}

impl Drop for ShaderLoader {
    fn drop(&mut self) {
        if let Some(ShaderReloader { watcher, join_handle }) = self.reloader.take() {
            drop(watcher);
            join_handle.join().unwrap();
        }
        for (_, shader) in self
            .new_shaders
            .lock()
            .unwrap()
            .drain()
            .chain(self.current_shaders.drain())
        {
            unsafe {
                self.context.device.destroy_shader_module(Some(shader), None);
            }
        }
    }
}

const MAX_DESCRIPTOR_SETS_PER_PIPELINE: usize = 4;

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineLayoutKey(ArrayVec<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS_PER_PIPELINE>);

struct PipelineLayoutCache {
    context: SharedContext,
    layouts: HashMap<PipelineLayoutKey, vk::PipelineLayout>,
}

impl PipelineLayoutCache {
    fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            layouts: HashMap::new(),
        }
    }

    fn get_layout(&mut self, descriptor_set_layouts: &[vk::DescriptorSetLayout]) -> vk::PipelineLayout {
        let device = &self.context.device;
        let key = PipelineLayoutKey(descriptor_set_layouts.try_into().unwrap());
        *self.layouts.entry(key).or_insert_with(|| {
            let create_info = vk::PipelineLayoutCreateInfo::builder().p_set_layouts(descriptor_set_layouts);
            unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap()
        })
    }
}

impl Drop for PipelineLayoutCache {
    fn drop(&mut self) {
        let device = &self.context.device;
        for (_, layout) in self.layouts.drain() {
            unsafe { device.destroy_pipeline_layout(Some(layout), None) };
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct GraphicsPipelineState {
    vertex_input_bindings: ArrayVec<vk::VertexInputBindingDescription, 4>,
    vertex_input_attributes: ArrayVec<vk::VertexInputAttributeDescription, 8>,
    topology: vk::PrimitiveTopology,
    render_pass: vk::RenderPass, // TODO: replace with state for *compatible* pass
    samples: vk::SampleCountFlags,
    depth_compare_op: vk::CompareOp,
}

impl GraphicsPipelineState {
    pub fn new(render_pass: vk::RenderPass, samples: vk::SampleCountFlags) -> Self {
        Self {
            vertex_input_bindings: ArrayVec::new(),
            vertex_input_attributes: ArrayVec::new(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            render_pass,
            samples,
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
        }
    }

    pub fn with_vertex_inputs(
        mut self,
        bindings: &[vk::VertexInputBindingDescription],
        attributes: &[vk::VertexInputAttributeDescription],
    ) -> Self {
        self.vertex_input_bindings.clear();
        self.vertex_input_bindings.try_extend_from_slice(bindings).unwrap();
        self.vertex_input_attributes.clear();
        self.vertex_input_attributes.try_extend_from_slice(attributes).unwrap();
        self
    }

    pub fn with_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn with_depth_compare_op(mut self, compare_op: vk::CompareOp) -> Self {
        self.depth_compare_op = compare_op;
        self
    }
}

pub enum RayTracingShaderGroupDesc<'a> {
    Raygen(&'a str),
    Miss(&'a str),
    Hit {
        closest_hit: &'a str,
        any_hit: Option<&'a str>,
        intersection: Option<&'a str>,
    },
    Callable(&'a str),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RayTracingShaderGroup {
    Raygen(vk::ShaderModule),
    Miss(vk::ShaderModule),
    Hit {
        closest_hit: vk::ShaderModule,
        any_hit: Option<vk::ShaderModule>,
        intersection: Option<vk::ShaderModule>,
    },
    Callable(vk::ShaderModule),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecializationConstantData {
    Bool32(vk::Bool32),
    U32(u32),
}

impl SpecializationConstantData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            SpecializationConstantData::Bool32(value) => bytemuck::bytes_of(value),
            SpecializationConstantData::U32(value) => bytemuck::bytes_of(value),
        }
    }
}
impl From<bool> for SpecializationConstantData {
    fn from(value: bool) -> Self {
        SpecializationConstantData::Bool32(if value { vk::TRUE } else { vk::FALSE })
    }
}
impl From<u32> for SpecializationConstantData {
    fn from(value: u32) -> Self {
        SpecializationConstantData::U32(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpecializationConstant {
    pub id: u32,
    pub data: SpecializationConstantData,
}

impl SpecializationConstant {
    pub fn new(id: u32, data: impl Into<SpecializationConstantData>) -> Self {
        Self { id, data: data.into() }
    }
}

struct SpecializationData {
    map_entries: Vec<vk::SpecializationMapEntry>,
    store: Vec<u8>,
}

impl SpecializationData {
    fn new(constants: &[SpecializationConstant]) -> Self {
        let mut map_entries = Vec::new();
        let mut store = Vec::new();
        for constant in constants {
            let bytes = constant.data.as_bytes();
            map_entries.push(vk::SpecializationMapEntry {
                constant_id: constant.id,
                offset: store.len() as u32,
                size: bytes.len(),
            });
            store.extend_from_slice(bytes);
        }
        Self { map_entries, store }
    }

    fn info(&self) -> <vk::SpecializationInfo as Builder>::Type {
        vk::SpecializationInfo::builder()
            .p_map_entries(&self.map_entries)
            .p_data(&self.store)
    }
}

pub enum VertexShaderDesc<'a> {
    Standard {
        vertex: &'a str,
        // TODO: tesellation/geometry shader names
    },
    Mesh {
        task: &'a str,
        task_constants: &'a [SpecializationConstant],
        task_subgroup_size: Option<u32>,
        mesh: &'a str,
        mesh_constants: &'a [SpecializationConstant],
    },
}

impl<'a> VertexShaderDesc<'a> {
    pub fn standard(vertex: &'a str) -> Self {
        Self::Standard { vertex }
    }

    pub fn mesh(
        task: &'a str,
        task_constants: &'a [SpecializationConstant],
        task_subgroup_size: Option<u32>,
        mesh: &'a str,
        mesh_constants: &'a [SpecializationConstant],
    ) -> Self {
        Self::Mesh {
            task,
            task_constants,
            task_subgroup_size,
            mesh,
            mesh_constants,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum VertexShaderKey {
    Standard {
        vertex: vk::ShaderModule,
    },
    Mesh {
        task: vk::ShaderModule,
        task_constants: Vec<SpecializationConstant>,
        task_subgroup_size: Option<u32>,
        mesh: vk::ShaderModule,
        mesh_constants: Vec<SpecializationConstant>,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, PartialEq, Eq, Hash)]
enum PipelineCacheKey {
    Compute {
        pipeline_layout: vk::PipelineLayout,
        shader: vk::ShaderModule,
        constants: Vec<SpecializationConstant>,
    },
    Graphics {
        pipeline_layout: vk::PipelineLayout,
        vertex_shader: VertexShaderKey,
        fragment_shader: vk::ShaderModule,
        state: GraphicsPipelineState,
    },
    Ui {
        render_pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
    },
    RayTracing {
        pipeline_layout: vk::PipelineLayout,
        shader_groups: Vec<RayTracingShaderGroup>,
    },
}

pub struct PipelineCache {
    context: SharedContext,
    shader_loader: ShaderLoader,
    layout_cache: RefCell<PipelineLayoutCache>,
    pipeline_cache: vk::PipelineCache,
    pipelines: RefCell<HashMap<PipelineCacheKey, vk::Pipeline>>,
}

impl PipelineCache {
    pub fn new<P: AsRef<Path>>(context: &SharedContext, path: P) -> Self {
        let layout_cache = PipelineLayoutCache::new(context);
        let pipeline_cache = {
            // TODO: load from file
            let create_info = vk::PipelineCacheCreateInfo {
                flags: if context
                    .physical_device_features
                    .pipeline_creation_cache_control
                    .pipeline_creation_cache_control
                    .as_bool()
                {
                    vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED_EXT
                } else {
                    vk::PipelineCacheCreateFlags::empty()
                },
                ..Default::default()
            };
            unsafe { context.device.create_pipeline_cache(&create_info, None) }.unwrap()
        };
        Self {
            context: SharedContext::clone(context),
            shader_loader: ShaderLoader::new(context, path),
            layout_cache: RefCell::new(layout_cache),
            pipeline_cache,
            pipelines: RefCell::new(HashMap::new()),
        }
    }

    pub fn begin_frame(&mut self) {
        self.shader_loader.transfer_new_shaders();
    }

    pub fn get_pipeline_layout(&self, descriptor_set_layouts: &[vk::DescriptorSetLayout]) -> vk::PipelineLayout {
        self.layout_cache.borrow_mut().get_layout(descriptor_set_layouts)
    }

    pub fn get_compute(
        &self,
        shader_name: &str,
        constants: &[SpecializationConstant],
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let shader = self.shader_loader.get_shader(shader_name).unwrap();
        let key = PipelineCacheKey::Compute {
            pipeline_layout,
            shader,
            constants: constants.to_vec(),
        };
        *self.pipelines.borrow_mut().entry(key).or_insert_with(|| {
            let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let specialization_data = SpecializationData::new(constants);
            let specialization_info = vk::SpecializationInfo::builder()
                .p_map_entries(&specialization_data.map_entries)
                .p_data(&specialization_data.store);
            let pipeline_create_info = vk::ComputePipelineCreateInfo {
                stage: vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module: Some(shader),
                    p_name: shader_entry_name.as_ptr(),
                    p_specialization_info: &*specialization_info,
                    ..Default::default()
                },
                layout: Some(pipeline_layout),
                ..Default::default()
            };
            unsafe {
                self.context.device.create_compute_pipelines_single(
                    Some(self.pipeline_cache),
                    &pipeline_create_info,
                    None,
                )
            }
            .unwrap()
        })
    }

    pub fn get_graphics(
        &self,
        vertex_shader_desc: VertexShaderDesc,
        fragment_shader_name: &str,
        pipeline_layout: vk::PipelineLayout,
        state: &GraphicsPipelineState,
    ) -> vk::Pipeline {
        let vertex_shader = match vertex_shader_desc {
            VertexShaderDesc::Standard { vertex } => VertexShaderKey::Standard {
                vertex: self.shader_loader.get_shader(vertex).unwrap(),
            },
            VertexShaderDesc::Mesh {
                task,
                task_constants,
                task_subgroup_size,
                mesh,
                mesh_constants,
            } => VertexShaderKey::Mesh {
                task: self.shader_loader.get_shader(task).unwrap(),
                task_constants: task_constants.to_vec(),
                task_subgroup_size,
                mesh: self.shader_loader.get_shader(mesh).unwrap(),
                mesh_constants: mesh_constants.to_vec(),
            },
        };
        let fragment_shader = self.shader_loader.get_shader(fragment_shader_name).unwrap();
        let key = PipelineCacheKey::Graphics {
            pipeline_layout,
            vertex_shader: vertex_shader.clone(),
            fragment_shader,
            state: state.clone(),
        };
        *self.pipelines.borrow_mut().entry(key).or_insert_with(|| {
            let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let mut specialization_data = ArrayVec::<SpecializationData, 2>::new();
            let mut specialization_info = ArrayVec::<vk::SpecializationInfo, 2>::new();
            let mut required_subgroup_size_create_info =
                ArrayVec::<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT, 1>::new();
            let mut shader_stage_create_info = ArrayVec::<vk::PipelineShaderStageCreateInfo, 3>::new();
            match vertex_shader {
                VertexShaderKey::Standard { vertex } => {
                    shader_stage_create_info.push(vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::VERTEX,
                        module: Some(vertex),
                        p_name: shader_entry_name.as_ptr(),
                        ..Default::default()
                    });
                }
                VertexShaderKey::Mesh {
                    task,
                    task_constants,
                    task_subgroup_size,
                    mesh,
                    mesh_constants,
                } => {
                    shader_stage_create_info.push({
                        specialization_data.push(SpecializationData::new(&task_constants));
                        specialization_info.push(*specialization_data.last().unwrap().info());
                        let mut p_next = ptr::null();
                        let mut flags = vk::PipelineShaderStageCreateFlags::empty();
                        if let Some(task_subgroup_size) = task_subgroup_size {
                            required_subgroup_size_create_info.push(
                                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT {
                                    required_subgroup_size: task_subgroup_size,
                                    ..Default::default()
                                },
                            );
                            p_next = required_subgroup_size_create_info.last().unwrap() as *const _ as *const _;
                            flags |= vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT;
                        };
                        vk::PipelineShaderStageCreateInfo {
                            p_next,
                            flags,
                            stage: vk::ShaderStageFlags::TASK_NV,
                            module: Some(task),
                            p_name: shader_entry_name.as_ptr(),
                            p_specialization_info: specialization_info.last().unwrap(),
                            ..Default::default()
                        }
                    });
                    shader_stage_create_info.push({
                        specialization_data.push(SpecializationData::new(&mesh_constants));
                        specialization_info.push(*specialization_data.last().unwrap().info());
                        vk::PipelineShaderStageCreateInfo {
                            stage: vk::ShaderStageFlags::MESH_NV,
                            module: Some(mesh),
                            p_name: shader_entry_name.as_ptr(),
                            p_specialization_info: specialization_info.last().unwrap(),
                            ..Default::default()
                        }
                    });
                }
            }
            shader_stage_create_info.push(vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: Some(fragment_shader),
                p_name: shader_entry_name.as_ptr(),
                ..Default::default()
            });

            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .p_vertex_attribute_descriptions(&state.vertex_input_attributes)
                .p_vertex_binding_descriptions(&state.vertex_input_bindings);
            let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: state.topology,
                ..Default::default()
            };

            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::BACK,
                line_width: 1.0,
                ..Default::default()
            };
            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: state.samples,
                ..Default::default()
            };

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(state.depth_compare_op);

            let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
                color_write_mask: vk::ColorComponentFlags::all(),
                ..Default::default()
            };
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .p_attachments(slice::from_ref(&color_blend_attachment_state));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let pipeline_dynamic_state_create_info =
                vk::PipelineDynamicStateCreateInfo::builder().p_dynamic_states(&dynamic_states);

            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
                .p_stages(&shader_stage_create_info)
                .p_vertex_input_state(Some(&vertex_input_state_create_info))
                .p_input_assembly_state(Some(&input_assembly_state_create_info))
                .p_viewport_state(Some(&viewport_state_create_info))
                .p_rasterization_state(Some(&rasterization_state_create_info))
                .p_multisample_state(Some(&multisample_state_create_info))
                .p_depth_stencil_state(Some(&depth_stencil_state))
                .p_color_blend_state(Some(&color_blend_state_create_info))
                .p_dynamic_state(Some(&pipeline_dynamic_state_create_info))
                .layout(Some(pipeline_layout))
                .render_pass(Some(state.render_pass));

            unsafe {
                self.context.device.create_graphics_pipelines_single(
                    Some(self.pipeline_cache),
                    &pipeline_create_info,
                    None,
                )
            }
            .unwrap()
        })
    }

    pub fn get_ui(
        &self,
        egui_renderer: &spark_egui::Renderer,
        render_pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
    ) -> vk::Pipeline {
        let key = PipelineCacheKey::Ui { render_pass, samples };
        *self
            .pipelines
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| egui_renderer.create_pipeline(&self.context.device, render_pass, samples))
    }

    pub fn get_ray_tracing(
        &self,
        group_desc: &[RayTracingShaderGroupDesc],
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let shader_groups: Vec<_> = group_desc
            .iter()
            .map(|desc| match desc {
                RayTracingShaderGroupDesc::Raygen(raygen) => {
                    RayTracingShaderGroup::Raygen(self.shader_loader.get_shader(raygen).unwrap())
                }
                RayTracingShaderGroupDesc::Miss(miss) => {
                    RayTracingShaderGroup::Miss(self.shader_loader.get_shader(miss).unwrap())
                }
                RayTracingShaderGroupDesc::Hit {
                    closest_hit,
                    any_hit,
                    intersection,
                } => RayTracingShaderGroup::Hit {
                    closest_hit: self.shader_loader.get_shader(closest_hit).unwrap(),
                    any_hit: any_hit.map(|name| self.shader_loader.get_shader(name).unwrap()),
                    intersection: intersection.map(|name| self.shader_loader.get_shader(name).unwrap()),
                },
                RayTracingShaderGroupDesc::Callable(callable) => {
                    RayTracingShaderGroup::Callable(self.shader_loader.get_shader(callable).unwrap())
                }
            })
            .collect();
        let key = PipelineCacheKey::RayTracing {
            pipeline_layout,
            shader_groups: shader_groups.clone(),
        };
        *self.pipelines.borrow_mut().entry(key).or_insert_with(|| {
            let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let mut shader_stage_create_info = Vec::new();
            let mut get_stage_index = |stage, module| {
                if let Some(i) = shader_stage_create_info.iter().enumerate().find_map(
                    |(i, info): (usize, &vk::PipelineShaderStageCreateInfo)| {
                        if stage == info.stage && Some(module) == info.module {
                            Some(i as u32)
                        } else {
                            None
                        }
                    },
                ) {
                    i
                } else {
                    shader_stage_create_info.push(vk::PipelineShaderStageCreateInfo {
                        stage,
                        module: Some(module),
                        p_name: shader_entry_name.as_ptr(),
                        ..Default::default()
                    });
                    (shader_stage_create_info.len() - 1) as u32
                }
            };

            let shader_group_create_info: Vec<_> = shader_groups
                .iter()
                .map(|group| match group {
                    RayTracingShaderGroup::Raygen(raygen) => vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                        general_shader: get_stage_index(vk::ShaderStageFlags::RAYGEN_KHR, *raygen),
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    },
                    RayTracingShaderGroup::Miss(miss) => vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                        general_shader: get_stage_index(vk::ShaderStageFlags::MISS_KHR, *miss),
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    },
                    RayTracingShaderGroup::Hit {
                        closest_hit,
                        any_hit,
                        intersection,
                    } => vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: if intersection.is_some() {
                            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                        } else {
                            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                        },
                        general_shader: vk::SHADER_UNUSED_KHR,
                        closest_hit_shader: get_stage_index(vk::ShaderStageFlags::CLOSEST_HIT_KHR, *closest_hit),
                        any_hit_shader: any_hit.map_or(vk::SHADER_UNUSED_KHR, |module| {
                            get_stage_index(vk::ShaderStageFlags::ANY_HIT_KHR, module)
                        }),
                        intersection_shader: intersection.map_or(vk::SHADER_UNUSED_KHR, |module| {
                            get_stage_index(vk::ShaderStageFlags::INTERSECTION_KHR, module)
                        }),
                        ..Default::default()
                    },
                    RayTracingShaderGroup::Callable(callable) => vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                        general_shader: get_stage_index(vk::ShaderStageFlags::CALLABLE_KHR, *callable),
                        closest_hit_shader: vk::SHADER_UNUSED_KHR,
                        any_hit_shader: vk::SHADER_UNUSED_KHR,
                        intersection_shader: vk::SHADER_UNUSED_KHR,
                        ..Default::default()
                    },
                })
                .collect();

            let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::builder()
                .p_stages(&shader_stage_create_info)
                .p_groups(&shader_group_create_info)
                .layout(pipeline_layout)
                .max_pipeline_ray_recursion_depth(0);

            unsafe {
                self.context.device.create_ray_tracing_pipelines_khr_single(
                    None,
                    Some(self.pipeline_cache),
                    &pipeline_create_info,
                    None,
                )
            }
            .unwrap()
        })
    }

    pub fn ui_stats_table_rows(&self, ui: &mut egui::Ui) {
        self.shader_loader.ui_stats_table_rows(ui);

        ui.label("pipeline");
        ui.label(format!("{}", self.pipelines.borrow_mut().len()));
        ui.end_row();
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        let device = &self.context.device;
        for (_, pipeline) in self.pipelines.borrow_mut().drain() {
            unsafe {
                device.destroy_pipeline(Some(pipeline), None);
            }
        }
        unsafe {
            // TODO: save to file
            device.destroy_pipeline_cache(Some(self.pipeline_cache), None)
        }
    }
}
