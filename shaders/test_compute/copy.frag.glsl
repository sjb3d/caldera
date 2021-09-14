#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "tone_map.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, scalar) uniform CopyData {
    ivec2 offset;
    uvec2 trace_dims;
    float trace_scale;
} g_copy;

layout(set = 0, binding = 1, r32f) uniform readonly image2D g_image[3];

void main()
{
    const ivec2 coord = ivec2(gl_FragCoord.xy) + g_copy.offset;
    vec3 col = vec3(0.f);
    if (all(lessThan(uvec2(coord), g_copy.trace_dims))) {
        col.x = imageLoad(g_image[0], coord).x*g_copy.trace_scale;
        col.y = imageLoad(g_image[1], coord).x*g_copy.trace_scale;
        col.z = imageLoad(g_image[2], coord).x*g_copy.trace_scale;
    }
    o_col = vec4(linear_from_gamma(filmic_tone_map(col)), 1.f);
}
