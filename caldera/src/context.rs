use crate::window_surface;
use spark::{vk, Builder, Device, DeviceExtensions, Instance, InstanceExtensions, Loader};
use std::ffi::CStr;
use std::os::raw::c_void;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use winit::window::Window;

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

    unsafe fn create_pipeline_layout_from_ref(
        &self,
        descriptor_set_layout: &vk::DescriptorSetLayout,
    ) -> spark::Result<vk::PipelineLayout>;
}

impl DeviceExt for Device {
    unsafe fn get_buffer_device_address_helper(&self, buffer: vk::Buffer) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo {
            buffer: Some(buffer),
            ..Default::default()
        };
        self.get_buffer_device_address(&info)
    }

    unsafe fn create_pipeline_layout_from_ref(
        &self,
        descriptor_set_layout: &vk::DescriptorSetLayout,
    ) -> spark::Result<vk::PipelineLayout> {
        let create_info = vk::PipelineLayoutCreateInfo::builder().p_set_layouts(slice::from_ref(descriptor_set_layout));
        self.create_pipeline_layout(&create_info, None)
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

pub struct ContextParams {
    pub version: vk::Version,
    pub is_debug: bool,
    pub enable_geometry_shader: bool,
    pub allow_inline_uniform_block: bool,
    pub allow_ray_tracing: bool,
}

impl ContextParams {
    pub fn parse_arg(&mut self, s: &str) -> bool {
        match s {
            "-d" => {
                self.is_debug = true;
                true
            }
            "--vk10" => {
                self.version = vk::Version::from_raw_parts(1, 0, 0);
                true
            }
            "--vk11" => {
                self.version = vk::Version::from_raw_parts(1, 1, 0);
                true
            }
            "--vk12" => {
                self.version = vk::Version::from_raw_parts(1, 2, 0);
                true
            }
            "--no-iub" => {
                self.allow_inline_uniform_block = false;
                true
            }
            _ => false,
        }
    }
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            version: Default::default(),
            is_debug: false,
            enable_geometry_shader: false,
            allow_inline_uniform_block: true,
            allow_ray_tracing: false,
        }
    }
}

pub struct ContextRayTracingPipelineProperties {
    pub shader_group_handle_size: u32,
    pub shader_group_base_alignment: u32,
    pub shader_group_handle_alignment: u32,
}

pub struct Context {
    pub instance: Instance,
    pub debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub ray_tracing_pipeline_properties: Option<ContextRayTracingPipelineProperties>,
    pub enable_buffer_device_addresses: bool,
    pub queue_family_index: u32,
    pub queue_family_properties: vk::QueueFamilyProperties,
    pub queue: vk::Queue,
    pub device: Device,
    pub next_handle_uid: AtomicU64,
}

impl Context {
    pub fn new(window: &Window, params: &ContextParams) -> Self {
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
            window_surface::enable_extensions(window, &mut extensions);
            if params.is_debug {
                extensions.enable_ext_debug_utils();
            }
            if params.allow_inline_uniform_block && available_extensions.supports_ext_inline_uniform_block() {
                extensions.enable_ext_inline_uniform_block();
            }
            if params.allow_ray_tracing
                && available_extensions.supports_khr_acceleration_structure()
                && available_extensions.supports_khr_ray_tracing_pipeline()
                && available_extensions.supports_khr_ray_query()
            {
                extensions.enable_khr_acceleration_structure();
                extensions.enable_khr_ray_tracing_pipeline();
                extensions.enable_khr_ray_query();
            }
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

        let debug_utils_messenger = if params.is_debug {
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

        let surface = window_surface::create(&instance, window).unwrap();

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

        let ray_tracing_pipeline_properties =
            if instance.extensions.core_version >= vk::Version::from_raw_parts(1, 1, 0) {
                let mut rtpp = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
                let mut properties2 = vk::PhysicalDeviceProperties2::builder().insert_next(&mut rtpp);
                unsafe { instance.get_physical_device_properties2(physical_device, properties2.as_mut()) };
                Some(ContextRayTracingPipelineProperties {
                    shader_group_handle_size: rtpp.shader_group_handle_size,
                    shader_group_base_alignment: rtpp.shader_group_base_alignment,
                    shader_group_handle_alignment: rtpp.shader_group_handle_alignment,
                })
            } else {
                None
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
                        && unsafe {
                            instance.get_physical_device_surface_support_khr(physical_device, index as u32, surface)
                        }
                        .unwrap()
                    {
                        Some((index as u32, *info))
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };

        let mut enable_ray_tracing = false;
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

            let mut enabled_features = vk::PhysicalDeviceFeatures::default();
            if params.enable_geometry_shader {
                enabled_features.geometry_shader = vk::TRUE;
            }

            let available_extensions = {
                let extension_properties =
                    unsafe { instance.enumerate_device_extension_properties_to_vec(physical_device, None) }.unwrap();
                DeviceExtensions::from_properties(params.version, &extension_properties)
            };

            let mut extensions = DeviceExtensions::new(params.version);
            extensions.enable_khr_swapchain();
            extensions.enable_ext_scalar_block_layout();
            if params.allow_inline_uniform_block && available_extensions.supports_ext_inline_uniform_block() {
                extensions.enable_ext_inline_uniform_block();
            }
            if params.allow_ray_tracing
                && available_extensions.supports_khr_acceleration_structure()
                && available_extensions.supports_khr_ray_tracing_pipeline()
                && available_extensions.supports_khr_ray_query()
            {
                extensions.enable_khr_acceleration_structure();
                extensions.enable_khr_ray_tracing_pipeline();
                extensions.enable_khr_ray_query();
                enable_ray_tracing = true;
            }
            let extension_names = extensions.to_name_vec();
            for &name in extension_names.iter() {
                println!("loading device extension {:?}", name);
            }

            let mut scalar_block_layout_features =
                vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder().scalar_block_layout(true);
            let mut buffer_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::builder().buffer_device_address(enable_ray_tracing);
            let mut acceleration_structure_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(enable_ray_tracing);
            let mut ray_tracing_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(enable_ray_tracing);

            let extension_name_ptrs: Vec<_> = extension_names.iter().map(|s| s.as_ptr()).collect();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .p_queue_create_infos(slice::from_ref(&device_queue_create_info))
                .pp_enabled_extension_names(&extension_name_ptrs)
                .p_enabled_features(Some(&enabled_features))
                .insert_next(&mut scalar_block_layout_features)
                .insert_next(&mut buffer_device_address_features)
                .insert_next(&mut acceleration_structure_features)
                .insert_next(&mut ray_tracing_pipeline_features);

            unsafe { instance.create_device(physical_device, &device_create_info, None, params.version) }.unwrap()
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Self {
            instance,
            debug_utils_messenger,
            surface,
            physical_device,
            physical_device_properties,
            physical_device_memory_properties,
            ray_tracing_pipeline_properties,
            enable_buffer_device_addresses: enable_ray_tracing,
            queue_family_index,
            queue_family_properties,
            queue,
            device,
            next_handle_uid: AtomicU64::new(0),
        }
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
            self.instance.destroy_surface_khr(Some(self.surface), None);
            if self.debug_utils_messenger.is_some() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.debug_utils_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
