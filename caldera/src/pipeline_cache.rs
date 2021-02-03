use crate::context::Context;
use arrayvec::ArrayVec;
use imgui::Ui;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use spark::{vk, Builder, Device};
use std::collections::HashMap;
use std::ffi::CStr;
use std::fs::File;
use std::io;
use std::io::Read;
use std::mem;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

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
    context: Arc<Context>,
    base_path: PathBuf,
    reloader: Option<ShaderReloader>,
    current_shaders: HashMap<PathBuf, vk::ShaderModule>,
    new_shaders: Arc<Mutex<HashMap<PathBuf, vk::ShaderModule>>>,
}

impl ShaderLoader {
    pub fn new<P: AsRef<Path>>(context: &Arc<Context>, base_path: P) -> Self {
        let base_path = base_path.as_ref().to_owned();

        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::watcher(tx, Duration::from_millis(500)).unwrap();
        watcher.watch(&base_path, RecursiveMode::Recursive).unwrap();

        let current_shaders = HashMap::new();
        let new_shaders = Arc::new(Mutex::new(HashMap::new()));

        let join_handle = thread::spawn({
            let context = Arc::clone(&context);
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
            context: Arc::clone(&context),
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

    pub fn ui_stats_table_rows(&self, ui: &Ui) {
        ui.text("shader");
        ui.next_column();
        ui.text(format!("{}", self.current_shaders.len()));
        ui.next_column();
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

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct GraphicsPipelineState {
    vertex_input_bindings: ArrayVec<[vk::VertexInputBindingDescription; 4]>,
    vertex_input_attributes: ArrayVec<[vk::VertexInputAttributeDescription; 8]>,
    render_pass: vk::RenderPass, // TODO: replace with state for *compatible* pass
    samples: vk::SampleCountFlags,
}

impl GraphicsPipelineState {
    pub fn new(render_pass: vk::RenderPass, samples: vk::SampleCountFlags) -> Self {
        Self {
            vertex_input_bindings: ArrayVec::new(),
            vertex_input_attributes: ArrayVec::new(),
            render_pass,
            samples,
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
}

pub enum RayTracingShaderGroupDesc<'a> {
    Raygen(&'a str),
    Miss(&'a str),
    TrianglesHit {
        closest_hit: &'a str,
        any_hit: Option<&'a str>,
        intersection: Option<&'a str>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RayTracingShaderGroup {
    Raygen(vk::ShaderModule),
    Miss(vk::ShaderModule),
    TrianglesHit {
        closest_hit: vk::ShaderModule,
        any_hit: Option<vk::ShaderModule>,
        intersection: Option<vk::ShaderModule>,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, PartialEq, Eq, Hash)]
enum PipelineCacheKey {
    Compute {
        pipeline_layout: vk::PipelineLayout,
        shader: vk::ShaderModule,
    },
    Graphics {
        pipeline_layout: vk::PipelineLayout,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
        state: GraphicsPipelineState,
    },
    Ui {
        render_pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
    },
    RayTracing {
        pipeline_layout: vk::PipelineLayout,
        shader_groups: ArrayVec<[RayTracingShaderGroup; PipelineCache::RAY_TRACING_MAX_SHADER_GROUPS]>,
    },
}

pub struct PipelineCache {
    context: Arc<Context>,
    shader_loader: ShaderLoader,
    pipeline_cache: vk::PipelineCache,
    current_pipelines: HashMap<PipelineCacheKey, vk::Pipeline>,
    new_pipelines: Mutex<HashMap<PipelineCacheKey, vk::Pipeline>>,
}

impl PipelineCache {
    const RAY_TRACING_MAX_MODULES: usize = 8;
    const RAY_TRACING_MAX_SHADER_GROUPS: usize = 5;

    pub fn new<P: AsRef<Path>>(context: &Arc<Context>, path: P) -> Self {
        let pipeline_cache = {
            // TODO: load from file
            let create_info = vk::PipelineCacheCreateInfo {
                flags: if context.device.extensions.supports_ext_pipeline_creation_cache_control() {
                    vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED_EXT
                } else {
                    vk::PipelineCacheCreateFlags::empty()
                },
                ..Default::default()
            };
            unsafe { context.device.create_pipeline_cache(&create_info, None) }.unwrap()
        };
        Self {
            context: Arc::clone(&context),
            shader_loader: ShaderLoader::new(context, path),
            pipeline_cache,
            current_pipelines: HashMap::new(),
            new_pipelines: Mutex::new(HashMap::new()),
        }
    }

    pub fn begin_frame(&mut self) {
        self.shader_loader.transfer_new_shaders();

        let mut new_pipelines = self.new_pipelines.lock().unwrap();
        for (k, v) in new_pipelines.drain() {
            self.current_pipelines.insert(k, v);
        }
    }

    pub fn get_compute(&self, shader_name: &str, pipeline_layout: vk::PipelineLayout) -> vk::Pipeline {
        let shader = self.shader_loader.get_shader(shader_name).unwrap();
        let key = PipelineCacheKey::Compute {
            pipeline_layout,
            shader,
        };
        self.current_pipelines.get(&key).copied().unwrap_or_else(|| {
            // TODO: create pipeline on worker thread, for now we just block on miss
            let mut new_pipelines = self.new_pipelines.lock().unwrap();
            *new_pipelines.entry(key).or_insert_with(|| {
                let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                let pipeline_create_info = vk::ComputePipelineCreateInfo {
                    stage: vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::COMPUTE,
                        module: Some(shader),
                        p_name: shader_entry_name.as_ptr(),
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
        })
    }

    pub fn get_graphics(
        &self,
        vertex_shader_name: &str,
        fragment_shader_name: &str,
        pipeline_layout: vk::PipelineLayout,
        state: &GraphicsPipelineState,
    ) -> vk::Pipeline {
        let vertex_shader = self.shader_loader.get_shader(vertex_shader_name).unwrap();
        let fragment_shader = self.shader_loader.get_shader(fragment_shader_name).unwrap();
        let key = PipelineCacheKey::Graphics {
            pipeline_layout,
            vertex_shader,
            fragment_shader,
            state: state.clone(),
        };
        self.current_pipelines.get(&key).copied().unwrap_or_else(|| {
            // TODO: create pipeline on worker thread, for now we just block on miss
            let mut new_pipelines = self.new_pipelines.lock().unwrap();
            *new_pipelines.entry(key).or_insert_with(|| {
                let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                let shader_stage_create_info = [
                    vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::VERTEX,
                        module: Some(vertex_shader),
                        p_name: shader_entry_name.as_ptr(),
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        module: Some(fragment_shader),
                        p_name: shader_entry_name.as_ptr(),
                        ..Default::default()
                    },
                ];

                let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                    .p_vertex_attribute_descriptions(&state.vertex_input_attributes)
                    .p_vertex_binding_descriptions(&state.vertex_input_bindings);
                let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
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
                    front_face: vk::FrontFace::CLOCKWISE,
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
                    .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

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
                    .p_rasterization_state(&rasterization_state_create_info)
                    .p_multisample_state(Some(&multisample_state_create_info))
                    .p_depth_stencil_state(Some(&depth_stencil_state))
                    .p_color_blend_state(Some(&color_blend_state_create_info))
                    .p_dynamic_state(Some(&pipeline_dynamic_state_create_info))
                    .layout(pipeline_layout)
                    .render_pass(state.render_pass);

                unsafe {
                    self.context.device.create_graphics_pipelines_single(
                        Some(self.pipeline_cache),
                        &pipeline_create_info,
                        None,
                    )
                }
                .unwrap()
            })
        })
    }

    pub fn get_ui(
        &self,
        ui_renderer: &spark_imgui::Renderer,
        render_pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
    ) -> vk::Pipeline {
        let key = PipelineCacheKey::Ui { render_pass, samples };
        self.current_pipelines.get(&key).copied().unwrap_or_else(|| {
            // TODO: create pipeline on worker thread, for now we just block on miss
            let mut new_pipelines = self.new_pipelines.lock().unwrap();
            *new_pipelines
                .entry(key)
                .or_insert_with(|| ui_renderer.create_pipeline(&self.context.device, render_pass, samples))
        })
    }

    pub fn get_ray_tracing(
        &self,
        group_desc: &[RayTracingShaderGroupDesc],
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        assert!(group_desc.len() <= Self::RAY_TRACING_MAX_SHADER_GROUPS);
        let shader_groups: ArrayVec<[_; Self::RAY_TRACING_MAX_SHADER_GROUPS]> = group_desc
            .iter()
            .map(|desc| match desc {
                RayTracingShaderGroupDesc::Raygen(raygen) => {
                    RayTracingShaderGroup::Raygen(self.shader_loader.get_shader(raygen).unwrap())
                }
                RayTracingShaderGroupDesc::Miss(miss) => {
                    RayTracingShaderGroup::Miss(self.shader_loader.get_shader(miss).unwrap())
                }
                RayTracingShaderGroupDesc::TrianglesHit {
                    closest_hit,
                    any_hit,
                    intersection,
                } => RayTracingShaderGroup::TrianglesHit {
                    closest_hit: self.shader_loader.get_shader(closest_hit).unwrap(),
                    any_hit: any_hit.map(|name| self.shader_loader.get_shader(name).unwrap()),
                    intersection: intersection.map(|name| self.shader_loader.get_shader(name).unwrap()),
                },
            })
            .collect();
        let key = PipelineCacheKey::RayTracing {
            pipeline_layout,
            shader_groups: shader_groups.clone(),
        };
        self.current_pipelines.get(&key).copied().unwrap_or_else(|| {
            // TODO: create pipeline on worker thread, for now we just block on miss
            let mut new_pipelines = self.new_pipelines.lock().unwrap();
            *new_pipelines.entry(key).or_insert_with(|| {
                let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                let mut shader_stage_create_info = ArrayVec::<[_; Self::RAY_TRACING_MAX_MODULES]>::new();
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

                let shader_group_create_info: ArrayVec<[_; Self::RAY_TRACING_MAX_SHADER_GROUPS]> = shader_groups
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
                        RayTracingShaderGroup::TrianglesHit {
                            closest_hit,
                            any_hit,
                            intersection,
                        } => vk::RayTracingShaderGroupCreateInfoKHR {
                            ty: vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
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
                    })
                    .collect();

                let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::builder()
                    .p_stages(&shader_stage_create_info)
                    .p_groups(&shader_group_create_info)
                    .layout(pipeline_layout)
                    .max_pipeline_ray_recursion_depth(1);

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
        })
    }

    pub fn ui_stats_table_rows(&self, ui: &imgui::Ui) {
        self.shader_loader.ui_stats_table_rows(ui);

        ui.text("pipeline");
        ui.next_column();
        ui.text(format!("{}", self.current_pipelines.len()));
        ui.next_column();
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        for (_, pipeline) in self
            .new_pipelines
            .lock()
            .unwrap()
            .drain()
            .chain(self.current_pipelines.drain())
        {
            unsafe {
                self.context.device.destroy_pipeline(Some(pipeline), None);
            }
        }
        unsafe {
            // TODO: save to file
            self.context
                .device
                .destroy_pipeline_cache(Some(self.pipeline_cache), None)
        }
    }
}
