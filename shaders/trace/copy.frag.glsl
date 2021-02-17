#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "tone_map.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, scalar) uniform CopyData {
    float exposure_scale;
    uint render_color_space;
    uint tone_map_method;
} g_copy;

layout(set = 0, binding = 1, rgba32f) uniform restrict readonly image2D g_result;

void main()
{
    vec4 result = imageLoad(g_result, ivec2(gl_FragCoord));
    vec3 col = max(result.xyz*(g_copy.exposure_scale/result.w), vec3(0.f));
    col = tone_map_sample(col, g_copy.render_color_space, g_copy.tone_map_method);
    o_col = vec4(col, 1.f);
}
