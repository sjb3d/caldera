# caldera

Vulkan and rust experiments. The code is split into a core `caldera` crate and a few different examples. Everything is work in progress and unstable, but this repository is public in case the code is interesting for others.

## Features

- Render graph implementation over Vulkan, for ease of use
  - Automatic memory allocation of temporary buffers and images
  - Automatic placement of barriers and layout transitions
- Makes use of [spark](https://github.com/sjb3d/spark) to manage Vulkan commands and extensions
- Live reload of shaders (not ray tracing pipeline shaders yet though)
- A procedural macro for descriptor set layouts
- Asynchronous loading support for static buffers and images

## Examples

Examples can be run as follows:

```
make && cargo run --example <example_name> -- --help
```

The call to `make` is required to build shaders, which depends on [glslangValidator](https://github.com/KhronosGroup/glslang).
On windows, [make for windows](http://gnuwin32.sourceforge.net/packages/make.htm) and the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) can provide these.
Omit `--help` and add other command-line arguments as necessary for each sample.

Please follow the link in the name of each example to show a more information about that example.

Screenshot | Description
--- | ---
[![compute image](images/test_compute.jpg)](examples/test_compute) | [**test_compute**](examples/test_compute)<br/>Initial test for synchronisation between compute and graphics.  Implements a toy path tracer in a single compute shader, reads the result during rasterization of the UI.
[![ray_tracing image](images/test_ray_tracing.jpg)](examples/test_ray_tracing) | [**test_ray_tracing**](examples/test_ray_tracing)<br/>Test of the `VK_KHR_acceleration_structure` and `VK_KHR_ray_tracing_pipeline` extensions.  Loads a PLY format mesh and draws a few instances using either rasterization or ray tracing.
[![mesh_shader image](images/test_ray_tracing.jpg)](examples/test_mesh_shader) | [**test_mesh_shader**](examples/test_mesh_shader)<br/>Test of the `NV_mesh_shader` extension.  Loads a PLY format mesh, makes some clusters, then draws the result using either the standard vertex pipeline or mesh imagesshaders.
[![living-room-2 image](images/path_tracer.jpg)](examples/path_tracer) | [**path_tracer**](examples/path_tracer)<br/>A spectral path tracer built on Vulkan ray tracing with support for several different surfaces and light types. The [README](examples/path_tracer) for this example contains many more details. The scene shown is from these [rendering resources](https://benedikt-bitterli.me/resources/) made available by Benedikt Bitterli.

## Procedural Macro for Descriptor Set Layout

The macro `descriptor_set_layout!` is implemented in `caldera-macro`. This allows the layout to be declared using struct-like syntax.  For example, consider the following bindings in GLSL:

```glsl
layout(set = 0, binding = 0, scalar) uniform CopyData {
    vec2 params;
    float more;
} g_copy;

layout(set = 0, binding = 1, r32f) uniform restrict image2D g_images[3];
```

The descriptor set layout for set 0 above can be generated using the macro (and the [bytemuck](https://crates.io/crates/bytemuck) crate) as follows:

```rust
// Use bytemuck::Pod to safely alias as bytes
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CopyData {
    params: Vec2, // [f32; 2] layout
    more: f32,
}

// Use caldera::descriptor_set_layout to generate a helper struct
descriptor_set_layout!(CopyDescriptorSetLayout {
    copy: UniformData<CopyData>,
    images: [StorageImage; 3],
});
```

This generates a `CopyDescriptorSetLayout` struct with two methods:

* A `new()` method that creates the corresponding Vulkan descriptor set layout, intended to be called once at startup.
* A `write()` method that fully writes a descriptor set with uniform data and buffer/image views, intended to be called each time the descriptor set needs (fully) writing each frame.

This helps to cut down on boilerplate code for descriptor sets that can be declared at build time.

## Potential Future Work

TODO

[ ] Buffer views?
[ ] Use futures for async loading
