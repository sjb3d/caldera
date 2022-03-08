use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use spark::{vk, Instance, InstanceExtensions, Result};
use winit::window::Window;

pub fn enable_extensions(window: &Window, extensions: &mut InstanceExtensions) {
    match window.raw_window_handle() {
        #[cfg(target_os = "linux")]
        RawWindowHandle::Xlib(..) => extensions.enable_khr_xlib_surface(),

        #[cfg(target_os = "windows")]
        RawWindowHandle::Win32(..) => extensions.enable_khr_win32_surface(),

        #[cfg(target_os = "android")]
        RawWindowHandle::AndroidNdk(..) => extensions.enable_khr_android_surface(),

        _ => unimplemented!(),
    }
}

pub fn create(instance: &Instance, window: &Window) -> Result<vk::SurfaceKHR> {
    match window.raw_window_handle() {
        #[cfg(target_os = "linux")]
        RawWindowHandle::Xlib(handle) => {
            let create_info = vk::XlibSurfaceCreateInfoKHR {
                dpy: handle.display as _,
                window: handle.window,
                ..Default::default()
            };
            unsafe { instance.create_xlib_surface_khr(&create_info, None) }
        }

        #[cfg(target_os = "windows")]
        RawWindowHandle::Win32(handle) => {
            let create_info = vk::Win32SurfaceCreateInfoKHR {
                hwnd: handle.hwnd,
                ..Default::default()
            };
            unsafe { instance.create_win32_surface_khr(&create_info, None) }
        }

        #[cfg(target_os = "android")]
        RawWindowHandle::AndroidNdk(handle) => {
            let create_info = vk::AndroidSurfaceCreateInfoKHR {
                window: handle.a_native_window as _,
                ..Default::default()
            };
            unsafe { instance.create_android_surface_khr(&create_info, None) }
        }

        _ => unimplemented!(),
    }
}
