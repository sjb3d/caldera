use crate::window_surface;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use spark::{vk, Builder, Device, DeviceExtensions, Instance, InstanceExtensions, Loader};
use std::{
    ffi::CStr,
    num,
    os::raw::c_void,
    slice,
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
};
use strum::{EnumString, EnumVariantNames};
use winit::window::Window;

pub trait AsBool {
    fn as_bool(self) -> bool;
}

impl AsBool for vk::Bool32 {
    fn as_bool(self) -> bool {
        self != vk::FALSE
    }
}

unsafe extern "system" fn debug_messenger(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    if let Some(data) = p_callback_data.as_ref() {
        let message = CStr::from_ptr(data.p_message);
        println!("{}, {}: {:?}", message_severity, message_types, message);
    }
    vk::FALSE
}

pub trait DeviceExt {
    unsafe fn get_buffer_device_address_helper(&self, buffer: vk::Buffer) -> vk::DeviceAddress;
}

impl DeviceExt for Device {
    unsafe fn get_buffer_device_address_helper(&self, buffer: vk::Buffer) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo {
            buffer: Some(buffer),
            ..Default::default()
        };
        self.get_buffer_device_address(&info)
    }
}

trait PhysicalDeviceMemoryPropertiesExt {
    fn types(&self) -> &[vk::MemoryType];
    fn heaps(&self) -> &[vk::MemoryHeap];
}

impl PhysicalDeviceMemoryPropertiesExt for vk::PhysicalDeviceMemoryProperties {
    fn types(&self) -> &[vk::MemoryType] {
        &self.memory_types[..self.memory_type_count as usize]
    }
    fn heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_heaps[..self.memory_heap_count as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unique<T>(pub T, u64);

impl<T> Unique<T> {
    pub fn new(obj: T, uid: u64) -> Self {
        Self(obj, uid)
    }
}

pub type UniqueBuffer = Unique<vk::Buffer>;
pub type UniqueImage = Unique<vk::Image>;
pub type UniqueImageView = Unique<vk::ImageView>;
pub type UniqueRenderPass = Unique<vk::RenderPass>;
pub type UniqueFramebuffer = Unique<vk::Framebuffer>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
pub enum ContextFeature {
    Disable,
    Optional,
    Require,
}

impl ContextFeature {
    fn apply(&self, is_supported: impl FnOnce() -> bool, enable_support: impl FnOnce(), on_error: impl FnOnce()) {
        match self {
            ContextFeature::Disable => {}
            ContextFeature::Optional => {
                if is_supported() {
                    enable_support();
                }
            }
            ContextFeature::Require => {
                if !is_supported() {
                    on_error();
                }
                enable_support();
            }
        }
    }
}

pub fn try_version_from_str(s: &str) -> Result<vk::Version, num::ParseIntError> {
    let mut parts = s.split('.');
    let major = parts.next().unwrap_or("").parse::<u32>()?;
    let minor = parts.next().unwrap_or("0").parse::<u32>()?;
    let patch = parts.next().unwrap_or("0").parse::<u32>()?;
    Ok(vk::Version::from_raw_parts(major, minor, patch))
}

pub struct ContextParams {
    pub version: vk::Version,
    pub debug_utils: ContextFeature,
    pub scalar_block_layout: ContextFeature,
    pub image_format_list: ContextFeature,
    pub pipeline_creation_cache_control: ContextFeature,
    pub geometry_shader: ContextFeature,
    pub inline_uniform_block: ContextFeature,
    pub bindless: ContextFeature,
    pub ray_tracing: ContextFeature,
    pub ray_query: ContextFeature,
    pub mesh_shader: ContextFeature,
    pub subgroup_size_control: ContextFeature,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            version: Default::default(),
            debug_utils: ContextFeature::Disable,
            scalar_block_layout: ContextFeature::Require,
            image_format_list: ContextFeature::Require,
            pipeline_creation_cache_control: ContextFeature::Disable,
            geometry_shader: ContextFeature::Disable,
            inline_uniform_block: ContextFeature::Optional,
            bindless: ContextFeature::Disable,
            ray_tracing: ContextFeature::Disable,
            ray_query: ContextFeature::Disable,
            mesh_shader: ContextFeature::Disable,
            subgroup_size_control: ContextFeature::Disable,
        }
    }
}

pub struct PhysicalDeviceExtraProperties {
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub mesh_shader: vk::PhysicalDeviceMeshShaderPropertiesNV,
    pub subgroup_size_control: vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT,
}

#[derive(Default)]
pub struct DeviceFeatures {
    pub base: vk::PhysicalDeviceFeatures,
    pub scalar_block_layout: vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT,
    pub pipeline_creation_cache_control: vk::PhysicalDevicePipelineCreationCacheControlFeaturesEXT,
    pub inline_uniform_block: vk::PhysicalDeviceInlineUniformBlockFeaturesEXT,
    pub buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    pub ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR,
    pub descriptor_indexing: vk::PhysicalDeviceDescriptorIndexingFeatures,
    pub mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesNV,
    pub subgroup_size_control: vk::PhysicalDeviceSubgroupSizeControlFeatures,
}

pub struct Context {
    pub instance: Instance,
    pub debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub surface: Option<vk::SurfaceKHR>,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub physical_device_extra_properties: Option<PhysicalDeviceExtraProperties>,
    pub physical_device_features: DeviceFeatures,
    pub queue_family_index: u32,
    pub queue_family_properties: vk::QueueFamilyProperties,
    pub queue: vk::Queue,
    pub device: Device,
    pub next_handle_uid: AtomicU64,
}

pub type SharedContext = Arc<Context>;

impl Context {
    pub fn new(window: Option<&Window>, params: &ContextParams) -> SharedContext {
        let instance = {
            let loader = Loader::new().unwrap();
            let instance_version = unsafe { loader.enumerate_instance_version() }.unwrap();
            println!(
                "loading instance version {} ({} supported)",
                params.version, instance_version
            );
            if instance_version < params.version {
                panic!(
                    "requested instance version {} is greater than the available version {}",
                    params.version, instance_version
                );
            }

            let available_extensions = {
                let extension_properties =
                    unsafe { loader.enumerate_instance_extension_properties_to_vec(None) }.unwrap();
                InstanceExtensions::from_properties(params.version, &extension_properties)
            };

            let mut extensions = InstanceExtensions::new(params.version);
            if let Some(window) = window {
                window_surface::enable_extensions(&window.raw_display_handle(), &mut extensions);
            }
            params.debug_utils.apply(
                || available_extensions.supports_ext_debug_utils(),
                || extensions.enable_ext_debug_utils(),
                || panic!("EXT_debug_utils not supported"),
            );
            params.scalar_block_layout.apply(
                || available_extensions.supports_ext_scalar_block_layout(),
                || extensions.enable_ext_scalar_block_layout(),
                || panic!("EXT_scalar_block_layout not supported"),
            );
            params.inline_uniform_block.apply(
                || available_extensions.supports_ext_inline_uniform_block(),
                || extensions.enable_ext_inline_uniform_block(),
                || panic!("EXT_inline_uniform_block not supported"),
            );
            params.bindless.apply(
                || available_extensions.supports_ext_descriptor_indexing(),
                || extensions.enable_ext_descriptor_indexing(),
                || panic!("EXT_descriptor_indexing not supported"),
            );
            params.ray_tracing.apply(
                || {
                    available_extensions.supports_khr_acceleration_structure()
                        && available_extensions.supports_khr_ray_tracing_pipeline()
                },
                || {
                    extensions.enable_khr_acceleration_structure();
                    extensions.enable_khr_ray_tracing_pipeline();
                },
                || panic!("KHR_acceleration_structure/KHR_ray_tracing_pipeline not supported"),
            );
            params.ray_query.apply(
                || {
                    available_extensions.supports_khr_acceleration_structure()
                        && available_extensions.supports_khr_ray_query()
                },
                || {
                    extensions.enable_khr_acceleration_structure();
                    extensions.enable_khr_ray_query();
                },
                || panic!("KHR_acceleration_structure/KHR_ray_query not supported"),
            );
            params.mesh_shader.apply(
                || available_extensions.supports_nv_mesh_shader(),
                || extensions.enable_nv_mesh_shader(),
                || panic!("NV_mesh_shader not supported"),
            );
            params.subgroup_size_control.apply(
                || available_extensions.supports_khr_get_physical_device_properties2(),
                || extensions.enable_khr_get_physical_device_properties2(),
                || panic!("KHR_get_physical_device_properties2 not supported"),
            );

            let extension_names = extensions.to_name_vec();
            for &name in extension_names.iter() {
                println!("loading instance extension {:?}", name);
            }

            let app_info = vk::ApplicationInfo::builder()
                .p_application_name(Some(CStr::from_bytes_with_nul(b"caldera\0").unwrap()))
                .api_version(params.version);

            let extension_name_ptrs: Vec<_> = extension_names.iter().map(|s| s.as_ptr()).collect();
            let instance_create_info = vk::InstanceCreateInfo::builder()
                .p_application_info(Some(&app_info))
                .pp_enabled_extension_names(&extension_name_ptrs);
            unsafe { loader.create_instance(&instance_create_info, None) }.unwrap()
        };

        let debug_utils_messenger = if instance.extensions.supports_ext_debug_utils() {
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(debug_messenger),
                ..Default::default()
            };
            Some(unsafe { instance.create_debug_utils_messenger_ext(&create_info, None) }.unwrap())
        } else {
            None
        };

        let surface = window.map(|window| {
            window_surface::create(&instance, &window.raw_display_handle(), &window.raw_window_handle()).unwrap()
        });

        let physical_device = {
            let physical_devices = unsafe { instance.enumerate_physical_devices_to_vec() }.unwrap();
            for physical_device in &physical_devices {
                let props = unsafe { instance.get_physical_device_properties(*physical_device) };
                println!("physical device ({}): {:?}", props.device_type, unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                });
            }
            physical_devices[0]
        };
        let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_version = physical_device_properties.api_version;

        let physical_device_extra_properties = if instance.extensions.supports_khr_get_physical_device_properties2() {
            let mut ray_tracing_pipeline = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut mesh_shader = vk::PhysicalDeviceMeshShaderPropertiesNV::default();
            let mut subgroup_size_control = vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::builder()
                .insert_next(&mut ray_tracing_pipeline)
                .insert_next(&mut mesh_shader)
                .insert_next(&mut subgroup_size_control);
            unsafe { instance.get_physical_device_properties2(physical_device, properties2.get_mut()) };
            Some(PhysicalDeviceExtraProperties {
                ray_tracing_pipeline,
                mesh_shader,
                subgroup_size_control,
            })
        } else {
            None
        };

        let available_features = if instance.extensions.supports_khr_get_physical_device_properties2() {
            let mut device_features = DeviceFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::builder()
                .insert_next(&mut device_features.scalar_block_layout)
                .insert_next(&mut device_features.pipeline_creation_cache_control)
                .insert_next(&mut device_features.inline_uniform_block)
                .insert_next(&mut device_features.buffer_device_address)
                .insert_next(&mut device_features.acceleration_structure)
                .insert_next(&mut device_features.ray_tracing_pipeline)
                .insert_next(&mut device_features.ray_query)
                .insert_next(&mut device_features.descriptor_indexing)
                .insert_next(&mut device_features.mesh_shader)
                .insert_next(&mut device_features.subgroup_size_control);
            unsafe { instance.get_physical_device_features2(physical_device, features2.get_mut()) };
            device_features.base = features2.features;
            device_features
        } else {
            DeviceFeatures {
                base: unsafe { instance.get_physical_device_features(physical_device) },
                ..Default::default()
            }
        };

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        for (i, mt) in physical_device_memory_properties.types().iter().enumerate() {
            println!("memory type {}: {}, heap {}", i, mt.property_flags, mt.heap_index);
        }
        for (i, mh) in physical_device_memory_properties.heaps().iter().enumerate() {
            println!("heap {}: {} bytes {}", i, mh.size, mh.flags);
        }

        let (queue_family_index, queue_family_properties) = {
            let queue_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;

            unsafe { instance.get_physical_device_queue_family_properties_to_vec(physical_device) }
                .iter()
                .enumerate()
                .filter_map(|(index, info)| {
                    if info.queue_flags.contains(queue_flags)
                        && surface
                            .map(|surface| {
                                unsafe {
                                    instance.get_physical_device_surface_support_khr(
                                        physical_device,
                                        index as u32,
                                        surface,
                                    )
                                }
                                .unwrap()
                            })
                            .unwrap_or(true)
                    {
                        Some((index as u32, *info))
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };

        let mut features = DeviceFeatures::default();
        let device = {
            println!(
                "loading device version {} ({} supported)",
                params.version, device_version
            );
            if device_version < params.version {
                panic!(
                    "requested device version {} is greater than the available version {}",
                    params.version, device_version
                );
            }

            let queue_priorities = [1.0];
            let device_queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .p_queue_priorities(&queue_priorities);

            let available_extensions = {
                let extension_properties =
                    unsafe { instance.enumerate_device_extension_properties_to_vec(physical_device, None) }.unwrap();
                DeviceExtensions::from_properties(params.version, &extension_properties)
            };

            let mut extensions = DeviceExtensions::new(params.version);

            if window.is_some() {
                extensions.enable_khr_swapchain();
            }
            params.geometry_shader.apply(
                || available_features.base.geometry_shader.as_bool(),
                || features.base.geometry_shader = vk::TRUE,
                || panic!("geometry shaders not supported"),
            );
            params.scalar_block_layout.apply(
                || {
                    available_extensions.supports_ext_scalar_block_layout()
                        && available_features.scalar_block_layout.scalar_block_layout.as_bool()
                },
                || {
                    extensions.enable_ext_scalar_block_layout();
                    features.scalar_block_layout.scalar_block_layout = vk::TRUE;
                },
                || panic!("EXT_scalar_block_layout not supported"),
            );
            params.image_format_list.apply(
                || available_extensions.supports_khr_image_format_list(),
                || extensions.enable_khr_image_format_list(),
                || panic!("KHR_image_format_list not supported"),
            );
            params.pipeline_creation_cache_control.apply(
                || {
                    available_extensions.supports_ext_pipeline_creation_cache_control()
                        && available_features
                            .pipeline_creation_cache_control
                            .pipeline_creation_cache_control
                            .as_bool()
                },
                || {
                    extensions.enable_ext_pipeline_creation_cache_control();
                    features.pipeline_creation_cache_control.pipeline_creation_cache_control = vk::TRUE;
                },
                || panic!("EXT_pipeline_creation_cache_control not support"),
            );
            params.inline_uniform_block.apply(
                || {
                    available_extensions.supports_ext_inline_uniform_block()
                        && available_features.inline_uniform_block.inline_uniform_block.as_bool()
                },
                || {
                    extensions.enable_ext_inline_uniform_block();
                    features.inline_uniform_block.inline_uniform_block = vk::TRUE;
                },
                || panic!("EXT_inline_uniform_block not supported"),
            );
            params.bindless.apply(
                || {
                    available_extensions.supports_ext_descriptor_indexing()
                        && available_features
                            .descriptor_indexing
                            .descriptor_binding_storage_buffer_update_after_bind
                            .as_bool()
                        && available_features
                            .descriptor_indexing
                            .descriptor_binding_sampled_image_update_after_bind
                            .as_bool()
                        && available_features
                            .descriptor_indexing
                            .shader_storage_buffer_array_non_uniform_indexing
                            .as_bool()
                        && available_features
                            .descriptor_indexing
                            .shader_sampled_image_array_non_uniform_indexing
                            .as_bool()
                        && available_features
                            .descriptor_indexing
                            .descriptor_binding_update_unused_while_pending
                            .as_bool()
                        && available_features
                            .descriptor_indexing
                            .descriptor_binding_partially_bound
                            .as_bool()
                },
                || {
                    extensions.enable_ext_descriptor_indexing();
                    features
                        .descriptor_indexing
                        .descriptor_binding_storage_buffer_update_after_bind = vk::TRUE;
                    features
                        .descriptor_indexing
                        .descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
                    features
                        .descriptor_indexing
                        .shader_storage_buffer_array_non_uniform_indexing = vk::TRUE;
                    features
                        .descriptor_indexing
                        .shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
                    features
                        .descriptor_indexing
                        .descriptor_binding_update_unused_while_pending = vk::TRUE;
                    features.descriptor_indexing.descriptor_binding_partially_bound = vk::TRUE;
                },
                || panic!("EXT_descriptor_indexing not supported"),
            );
            params.ray_tracing.apply(
                || {
                    available_extensions.supports_khr_ray_tracing_pipeline()
                        && available_features.buffer_device_address.buffer_device_address.as_bool()
                        && available_features.base.shader_int64.as_bool()
                        && available_features
                            .acceleration_structure
                            .acceleration_structure
                            .as_bool()
                        && available_features.ray_tracing_pipeline.ray_tracing_pipeline.as_bool()
                },
                || {
                    extensions.enable_khr_ray_tracing_pipeline();
                    features.buffer_device_address.buffer_device_address = vk::TRUE;
                    features.base.shader_int64 = vk::TRUE;
                    features.acceleration_structure.acceleration_structure = vk::TRUE;
                    features.ray_tracing_pipeline.ray_tracing_pipeline = vk::TRUE;
                },
                || panic!("KHR_acceleration_structure/KHR_ray_tracing_pipeline not supported"),
            );
            params.ray_query.apply(
                || {
                    available_extensions.supports_khr_ray_query()
                        && available_features.buffer_device_address.buffer_device_address.as_bool()
                        && available_features.base.shader_int64.as_bool()
                        && available_features
                            .acceleration_structure
                            .acceleration_structure
                            .as_bool()
                        && available_features.ray_query.ray_query.as_bool()
                },
                || {
                    extensions.enable_khr_ray_query();
                    features.buffer_device_address.buffer_device_address = vk::TRUE;
                    features.base.shader_int64 = vk::TRUE;
                    features.acceleration_structure.acceleration_structure = vk::TRUE;
                    features.ray_query.ray_query = vk::TRUE;
                },
                || panic!("KHR_acceleration_structure/KHR_ray_query not supported"),
            );
            params.mesh_shader.apply(
                || {
                    available_extensions.supports_nv_mesh_shader()
                        && available_features.mesh_shader.task_shader.as_bool()
                        && available_features.mesh_shader.mesh_shader.as_bool()
                },
                || {
                    extensions.enable_nv_mesh_shader();
                    features.mesh_shader.task_shader = vk::TRUE;
                    features.mesh_shader.mesh_shader = vk::TRUE;
                },
                || panic!("NV_mesh_shader not supported"),
            );
            params.subgroup_size_control.apply(
                || {
                    available_extensions.supports_ext_subgroup_size_control()
                        && available_features.subgroup_size_control.subgroup_size_control.as_bool()
                        && available_features
                            .subgroup_size_control
                            .compute_full_subgroups
                            .as_bool()
                },
                || {
                    extensions.enable_ext_subgroup_size_control();
                    features.subgroup_size_control.subgroup_size_control = vk::TRUE;
                    features.subgroup_size_control.compute_full_subgroups = vk::TRUE;
                },
                || panic!("EXT_subgroup_size_control not supported"),
            );

            let extension_names = extensions.to_name_vec();
            for &name in extension_names.iter() {
                println!("loading device extension {:?}", name);
            }

            let extension_name_ptrs: Vec<_> = extension_names.iter().map(|s| s.as_ptr()).collect();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .p_queue_create_infos(slice::from_ref(&device_queue_create_info))
                .pp_enabled_extension_names(&extension_name_ptrs)
                .p_enabled_features(Some(&features.base))
                .insert_next(&mut features.scalar_block_layout)
                .insert_next(&mut features.pipeline_creation_cache_control)
                .insert_next(&mut features.inline_uniform_block)
                .insert_next(&mut features.buffer_device_address)
                .insert_next(&mut features.acceleration_structure)
                .insert_next(&mut features.ray_tracing_pipeline)
                .insert_next(&mut features.ray_query)
                .insert_next(&mut features.descriptor_indexing)
                .insert_next(&mut features.mesh_shader)
                .insert_next(&mut features.subgroup_size_control);

            unsafe { instance.create_device(physical_device, &device_create_info, None, params.version) }.unwrap()
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        SharedContext::new(Self {
            instance,
            debug_utils_messenger,
            surface,
            physical_device,
            physical_device_properties,
            physical_device_memory_properties,
            physical_device_extra_properties,
            physical_device_features: features,
            queue_family_index,
            queue_family_properties,
            queue,
            device,
            next_handle_uid: AtomicU64::new(0),
        })
    }

    pub fn allocate_handle_uid(&self) -> u64 {
        self.next_handle_uid.fetch_add(1, Ordering::SeqCst)
    }

    pub fn get_memory_type_index(&self, type_filter: u32, property_flags: vk::MemoryPropertyFlags) -> Option<u32> {
        for (i, mt) in self.physical_device_memory_properties.types().iter().enumerate() {
            let i = i as u32;
            if (type_filter & (1 << i)) != 0 && mt.property_flags.contains(property_flags) {
                return Some(i);
            }
        }
        None
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if let Some(surface) = self.surface {
                self.instance.destroy_surface_khr(Some(surface), None);
            }
            if self.debug_utils_messenger.is_some() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.debug_utils_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
