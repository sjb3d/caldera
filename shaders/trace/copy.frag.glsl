#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"
#include "normal_pack.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, scalar) uniform CopyData {
    float sample_scale;
} g_copy;

layout(set = 0, binding = 1, r32f) uniform restrict readonly image2D g_result_r;
layout(set = 0, binding = 2, r32f) uniform restrict readonly image2D g_result_g;
layout(set = 0, binding = 3, r32f) uniform restrict readonly image2D g_result_b;

void main()
{
    const vec3 result = vec3(
        imageLoad(g_result_r, ivec2(gl_FragCoord.xy)).x,
        imageLoad(g_result_g, ivec2(gl_FragCoord.xy)).x,
        imageLoad(g_result_b, ivec2(gl_FragCoord.xy)).x)*g_copy.sample_scale;

    o_col = vec4(result, 1.f);
}
