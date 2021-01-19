# caldera

## Overview

Vulkan and rust experiments, everything is work in progress, in the areas of:

* Iterating on how [spark](https://github.com/sjb3d/spark) manages Vulkan commands and extensions
* A procedural macro for descriptor set layouts
* Render graph implementation, working but very inoptimal support for:
  * Automatic memory allocation of temporary buffers and images
  * Automatic placement of barriers and layout transitions
* Various helpers to cache Vulkan objects, with live reload of shaders
* Asynchronous loading of static buffers and images from the CPU

## Test Apps

### compute

TODO document (simple compute path tracer, PMJ samples, AcesCG color space)

### mesh

TODO document (load a PLY mesh, render with MSAA and transient attachments)

## Feature Details

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

### Render Graph Details

TODO document some details
