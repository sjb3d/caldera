#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(set = 0, binding = 1, scalar) uniform HashTableUniforms {
    HashTableInfo info;
} g_uniforms;
layout(set = 0, binding = 2, scalar) restrict buffer AgeHistogram {
    uint arr[];
} g_age_histogram;

layout(location = 0) out vec4 o_col;

layout(location = 0) in vec2 v_uv;

void main()
{
    float column_index_f = v_uv.x * float(MAX_AGE);
    uint column_index = min(uint(column_index_f), MAX_AGE - 1);
    float horiz_f = column_index_f - float(column_index);
    float vert_f = mix(1.05f, 0.f, v_uv.y);

    uint max_value = 0;
    uint display_value = 0;
    for (uint i = 0; i < MAX_AGE; ++i) {
        uint value = g_age_histogram.arr[i];
        if (i == column_index) {
            display_value = value;
        }
        max_value = max(max_value, value);
    }

    vec3 col = vec3(1.f);
    if (abs(horiz_f - .5f) < .4f) {
        if (vert_f < float(display_value)/float(max_value)) {
            col = vec3(.3f, .6f, .9f);
        }
    }    
    o_col = vec4(col, 1.f);
}
