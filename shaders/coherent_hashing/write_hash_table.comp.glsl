#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, scalar) uniform HashTableUniforms {
    HashTableInfo info;
} g_uniforms;
layout(set = 0, binding = 1, scalar) restrict buffer Entries {
    uint arr[];
} g_entries;
layout(set = 0, binding = 2, scalar) restrict buffer MaxAges {
    uint arr[];
} g_max_ages;

#define HASH_TABLE_INFO             g_uniforms.info
#define HASH_TABLE_ENTRIES_WRITE    g_entries.arr
#define HASH_TABLE_MAX_AGES         g_max_ages.arr
#include "hash_table_read_write.glsl"

layout(set = 0, binding = 3, r8) uniform restrict readonly image2D g_image;

void main()
{
    uvec2 coord = gl_GlobalInvocationID.xy;

    uint key = morton2d(coord);
    uint data = uint(clamp(int(imageLoad(g_image, ivec2(coord)).x*255.f + .5f), 0, 255));
    if (data != 0) {
        insert_entry(key, data);
    }
}
