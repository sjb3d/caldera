#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "color_space.glsl"

layout(location = 0) out vec4 o_col;

layout(std430, set = 0, binding = 0) uniform CopyData {
    ivec2 offset;
    uvec2 trace_dims;
    float trace_scale;
} g_copy;

layout(set = 0, binding = 1, r32f) uniform readonly image2D g_image_r;
layout(set = 0, binding = 2, r32f) uniform readonly image2D g_image_g;
layout(set = 0, binding = 3, r32f) uniform readonly image2D g_image_b;

void main()
{
    const ivec2 coord = ivec2(gl_FragCoord.xy) + g_copy.offset;
    vec3 col = vec3(0.f);
    if (all(lessThan(uvec2(coord), g_copy.trace_dims))) {
        col.x = imageLoad(g_image_r, coord).x*g_copy.trace_scale;
        col.y = imageLoad(g_image_g, coord).x*g_copy.trace_scale;
        col.z = imageLoad(g_image_b, coord).x*g_copy.trace_scale;
    }
    col = rec709_from_fit(odt_and_rrt_fit(rrt_sat(col)));
    o_col = vec4(col, 1.f);
}
