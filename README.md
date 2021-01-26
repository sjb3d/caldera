# caldera

## Overview

Vulkan and rust experiments, everything is work in progress, in the areas of:

* Iterating on how [spark](https://github.com/sjb3d/spark) manages Vulkan commands and extensions
* A procedural macro for descriptor set layouts
* Render graph implementation, working but very inoptimal support for:
  * Automatic memory allocation of temporary buffers and images
  * Automatic placement of barriers and layout transitions
* Various helpers to cache Vulkan objects, with live reload of shaders
  * Supported for vertex/fragment/compute but not yet for ray tracing pipeline shaders
* Asynchronous loading of static buffers and images from the CPU

## Test Apps

Shaders are currently built using make and [glslangValidator](https://github.com/KhronosGroup/glslang) (using [make for windows](http://gnuwin32.sourceforge.net/packages/make.htm) and the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) on Windows).

Test apps can be run using:

```
make && cargo run --bin <app_name>
```

### compute

![compute](https://github.com/sjb3d/caldera/blob/main/docs/compute.jpg)

A simple path tracer in a compute shader, also for tinkering with:

* [Progressive Multi-Jittered Sample Sequences](https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/) implemented in [pmj](https://github.com/sjb3d/pmj)
  * Several sequences are generated into a texture at startup and used tiled over the image
* Wide colour gamut in the [ACEScg](https://en.wikipedia.org/wiki/Academy_Color_Encoding_System) colour space, converted back to Rec709 using the fit from [BakingLab](https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl)
  * There are some derivations for these matrices in [`color_space.rs`](https://github.com/sjb3d/caldera/blob/main/apps/compute/src/color_space.rs), but the tonemap curve fit is used as-is

### mesh

![mesh](https://github.com/sjb3d/caldera/blob/main/docs/mesh.jpg)

Test project for `VK_KHR_acceleration_structure` and `VK_KHR_ray_tracing_pipeline`.  Takes a PLY mesh filename as argument ([Stanford bunny](http://graphics.stanford.edu/data/3Dscanrep/) shown above).

Has code for:
* Loading a PLY mesh using [ply-rs](https://github.com/Fluci/ply-rs)
* Basic rasterisation with instancing and MSAA support
  * Using Vulkan transient attachments for depth (and pre-resolve colour when using MSAA)
* Acceleration structure creation
  * A single bottom level acceleration structure for the PLY mesh
  * A top level acceleration structure that instances it a few times
* Simple ray tracing pipeline
  * Binds the index and vertex attribute buffers using a `shaderRecordEXT` block in the shader binding table, to be able to interpolate a vertex normal on hit

## Library Details

### Procedural Macro for Descriptor Set Layout

The macro `descriptor_set_layout!` is implemented in `caldera-macro`, and used in the test apps.  This allows the layout to be declared using struct-like syntax, here is an example:

```rust
descriptor_set_layout!(CopyDescriptorSetLayout {
    copy: UniformData<CopyData>,
    image_r: StorageImage,
    image_g: StorageImage,
    image_b: StorageImage,
});
```

This generates a `CopyDescriptorSetLayout` struct with two methods:

* A `new()` method that creates the corresponding Vulkan descriptor set layout, intended to be called once at startup.
* A `write()` method that fully writes a descriptor set with uniform data and buffer/image views, intended to be called each time the descriptor set needs (fully) writing each frame.

This helps to cut down on boilerplate code for descriptor sets that can be declared at build time.

### Render Graph Details

_TODO: figure out what bits are worth documenting_
