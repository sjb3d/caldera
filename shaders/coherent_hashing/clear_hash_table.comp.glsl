#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, scalar) uniform ClearHashTableUniforms {
    uint entry_count;
} g_clear;

layout(set = 0, binding = 1, scalar) restrict writeonly buffer Entries {
    uint arr[];
} g_entries;

// TODO: max age buffer

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index < g_clear.entry_count) {
        g_entries.arr[index] = 0;
    }
}
