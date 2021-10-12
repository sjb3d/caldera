// requires BINDLESS_SET_INDEX to be defined before inclusion

#define BINDLESS_MAX_STORAGE_BUFFERS        (16 * 1024)
#define BINDLESS_MAX_SAMPLED_IMAGE_2D       1024
#define BINDLESS_MAX_SAMPLERS               32

layout(set = BINDLESS_SET_INDEX, binding = 0, scalar) restrict readonly buffer BindlessBuf {
    uint arr[];
} g_storage_buffers[BINDLESS_MAX_STORAGE_BUFFERS];

layout(set = BINDLESS_SET_INDEX, binding = 1) uniform texture2D g_textures[BINDLESS_MAX_SAMPLED_IMAGE_2D];

layout(set = BINDLESS_SET_INDEX, binding = 2) uniform sampler g_samplers[BINDLESS_MAX_SAMPLERS];
