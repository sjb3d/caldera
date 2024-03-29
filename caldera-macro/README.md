# caldera-macro

## Features

This crate implements the macro `descriptor_set`, which can be used to declare Vulkan descriptor sets with struct-like syntax.

The struct members can use:

- `UniformData<T>` for a uniform buffer of type `T`
- `Sampler` for a sampler type
- `SampledImage` for sampled image views
- `CombinedImageSampler` to combine both of the above
- `StorageImage` or `[StorageImage; N]` for storage image views
- `StorageBuffer` for buffers
- `AccelerationStructure` for acceleration structure handles

These are allocated to sequential binding slots in a single descriptor set layout.

## Example

For example, consider the following bindings in GLSL:

```glsl
layout(set = 0, binding = 0, scalar) uniform CopyData {
    vec2 params;
    float more;
} g_copy;

layout(set = 0, binding = 1, r32f) uniform restrict image2D g_images[3];
```

The descriptor set for set 0 above can be generated using the macro (and the [bytemuck](https://crates.io/crates/bytemuck) crate) as follows:

```rust
// Use bytemuck::Pod to safely alias as bytes
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct CopyData {
    params: Vec2, // [f32; 2] layout
    more: f32,
}

// Use caldera::descriptor_set to generate a helper struct
descriptor_set!(CopyDescriptorSet {
    copy: UniformData<CopyData>,
    images: [StorageImage; 3],
});
```

This generates a `CopyDescriptorSet` struct with two methods:

* A `create()` method that looks up a cached layout then fully writes a descriptor set with uniform data and buffer/image views, intended to be called each time the descriptor set needs (fully) writing each frame.
* A `layout()` method that can optionally be called just to get the cached layout.

This helps to cut down on boilerplate code for descriptor sets that can be declared at build time. See any of the [caldera](https://github.com/sjb3d/caldera) examples for more examples of this macro in use.
