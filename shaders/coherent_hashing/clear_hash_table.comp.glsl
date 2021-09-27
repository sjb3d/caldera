#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, scalar) uniform HashTableUniforms {
    HashTableInfo info;
} g_uniforms;
layout(set = 0, binding = 1, scalar) restrict writeonly buffer Entries {
    uint arr[];
} g_entries;
layout(set = 0, binding = 2, scalar) restrict writeonly buffer MaxAges {
    uint arr[];
} g_max_ages;

// TODO: max age buffer

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index < g_uniforms.info.entry_count) {
        g_entries.arr[index] = 0;
        if (g_uniforms.info.store_max_age != 0) {
            g_max_ages.arr[index] = 0;
        }
    }
}
