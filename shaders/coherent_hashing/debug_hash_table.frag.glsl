#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(set = 0, binding = 1, scalar) uniform HashTableUniforms {
    HashTableInfo info;
} g_uniforms;
layout(set = 0, binding = 2, scalar) restrict readonly buffer Entries {
    uint arr[];
} g_entries;

layout(location = 0) out vec4 o_col;

layout(location = 0) in vec2 v_uv;

const uint DEBUG_WIDTH = 128;
const uint DEBUG_HEIGHT = 1024;

void main()
{
    uvec2 entry_coord = uvec2(v_uv*vec2(DEBUG_WIDTH, DEBUG_HEIGHT) + .5f);
    uint entry_index = entry_coord.y*DEBUG_WIDTH + entry_coord.x;
    vec3 col = vec3(0.f);
    if (entry_index < g_uniforms.info.entry_count) {
        Entry entry = make_entry(g_entries.arr[entry_index]);
        if (get_age(entry) != 0) {
            uvec2 image_coord = unmorton2d(get_key(entry));
            vec2 image_uv = vec2(image_coord)/1024.f;
            float s = float(get_data(entry))/255.f;
            col = mix(vec3(1.f), vec3(image_uv, 0.f), s);
        }
    } else {
        discard;
    }
    o_col = vec4(col, 1.f);
}
