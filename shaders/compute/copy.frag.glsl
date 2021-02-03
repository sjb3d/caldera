#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "color_space.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, scalar) uniform CopyData {
    ivec2 offset;
    uvec2 trace_dims;
    float trace_scale;
    uint render_color_space;
    uint tone_map_method;
} g_copy;

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

#define TONE_MAP_METHOD_NONE        0
#define TONE_MAP_METHOD_FILMIC_SRGB 1
#define TONE_MAP_METHOD_ACES_FIT    2

layout(set = 0, binding = 1, r32f) uniform readonly image2D g_image_r;
layout(set = 0, binding = 2, r32f) uniform readonly image2D g_image_g;
layout(set = 0, binding = 3, r32f) uniform readonly image2D g_image_b;

vec3 acescg_from_sample(vec3 c)
{
    switch (g_copy.render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return acescg_from_rec709(c);
        case RENDER_COLOR_SPACE_ACESCG: return c;
    }
}
vec3 rec709_from_sample(vec3 c)
{
    switch (g_copy.render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return c;
        case RENDER_COLOR_SPACE_ACESCG: return rec709_from_acescg(c);
    }
}

vec3 tone_map_sample(vec3 c)
{
    switch (g_copy.tone_map_method) {
        default:
        case TONE_MAP_METHOD_NONE: {
            return rec709_from_sample(c);
        }

        case TONE_MAP_METHOD_FILMIC_SRGB: {
            const vec3 src = rec709_from_sample(c);
            return linear_from_gamma(filmic_tone_map(src));
        }

        case TONE_MAP_METHOD_ACES_FIT: {
            const float exposure_adjust_to_balance_comparisons = 1.8f;
            const vec3 src = acescg_from_sample(c)*exposure_adjust_to_balance_comparisons;
            return rec709_from_fit(odt_and_rrt_fit(rrt_sat(src)));
        }
    }
}

void main()
{
    const ivec2 coord = ivec2(gl_FragCoord.xy) + g_copy.offset;
    vec3 col = vec3(0.f);
    if (all(lessThan(uvec2(coord), g_copy.trace_dims))) {
        col.x = imageLoad(g_image_r, coord).x*g_copy.trace_scale;
        col.y = imageLoad(g_image_g, coord).x*g_copy.trace_scale;
        col.z = imageLoad(g_image_b, coord).x*g_copy.trace_scale;
    }
    col = tone_map_sample(col);
    o_col = vec4(col, 1.f);
}
