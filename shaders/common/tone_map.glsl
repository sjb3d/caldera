#include "color_space.glsl"

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

#define TONE_MAP_METHOD_NONE        0
#define TONE_MAP_METHOD_FILMIC_SRGB 1
#define TONE_MAP_METHOD_ACES_FIT    2

#define ACES_EXPOSURE_ADJUST_TO_BALANCE     1.8f

vec3 acescg_from_sample(vec3 c, uint render_color_space)
{
    switch (render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return acescg_from_rec709(c);
        case RENDER_COLOR_SPACE_ACESCG: return c;
    }
}
vec3 rec709_from_sample(vec3 c, uint render_color_space)
{
    switch (render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return c;
        case RENDER_COLOR_SPACE_ACESCG: return rec709_from_acescg(c);
    }
}

vec3 sample_from_rec709(vec3 c, uint render_color_space)
{
    switch (render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return c;
        case RENDER_COLOR_SPACE_ACESCG: return acescg_from_rec709(c);
    }
}

vec3 tone_map_sample(vec3 c, uint render_color_space, uint tone_map_method)
{
    switch (tone_map_method) {
        default:
        case TONE_MAP_METHOD_NONE: {
            return rec709_from_sample(c, render_color_space);
        }

        case TONE_MAP_METHOD_FILMIC_SRGB: {
            const vec3 src = rec709_from_sample(c, render_color_space);
            return linear_from_gamma(filmic_tone_map(src));
        }

        case TONE_MAP_METHOD_ACES_FIT: {
            const vec3 src = acescg_from_sample(c, render_color_space)*ACES_EXPOSURE_ADJUST_TO_BALANCE;
            return rec709_from_fit(odt_and_rrt_fit(rrt_sat(src)));
        }
    }
}
vec3 tone_map_sample(vec3 c, mat3 rec709_from_xyz, mat3 acescg_from_xyz, uint tone_map_method)
{
    switch (tone_map_method) {
        default:
        case TONE_MAP_METHOD_NONE: {
            return rec709_from_xyz*c;
        }

        case TONE_MAP_METHOD_FILMIC_SRGB: {
            const vec3 src = rec709_from_xyz*c;
            return linear_from_gamma(filmic_tone_map(src));
        }

        case TONE_MAP_METHOD_ACES_FIT: {
            const vec3 src = (acescg_from_xyz*c)*ACES_EXPOSURE_ADJUST_TO_BALANCE;
            return rec709_from_fit(odt_and_rrt_fit(rrt_sat(src)));
        }
    }
}

vec3 tone_map_sample_to_gamma(vec3 c, uint render_color_space, uint tone_map_method)
{
    switch (tone_map_method) {
        default:
        case TONE_MAP_METHOD_NONE: {
            return gamma_from_linear(rec709_from_sample(c, render_color_space));
        }

        case TONE_MAP_METHOD_FILMIC_SRGB: {
            const vec3 src = rec709_from_sample(c, render_color_space);
            return filmic_tone_map(src);
        }

        case TONE_MAP_METHOD_ACES_FIT: {
            const vec3 src = acescg_from_sample(c, render_color_space)*ACES_EXPOSURE_ADJUST_TO_BALANCE;
            return gamma_from_linear(rec709_from_fit(odt_and_rrt_fit(rrt_sat(src))));
        }
    }
}
vec3 tone_map_sample_to_gamma(vec3 c, mat3 rec709_from_xyz, mat3 acescg_from_xyz, uint tone_map_method)
{
    switch (tone_map_method) {
        default:
        case TONE_MAP_METHOD_NONE: {
            return gamma_from_linear(rec709_from_xyz*c);
        }

        case TONE_MAP_METHOD_FILMIC_SRGB: {
            const vec3 src = rec709_from_xyz*c;
            return filmic_tone_map(src);
        }

        case TONE_MAP_METHOD_ACES_FIT: {
            const vec3 src = (acescg_from_xyz*c)*ACES_EXPOSURE_ADJUST_TO_BALANCE;
            return gamma_from_linear(rec709_from_fit(odt_and_rrt_fit(rrt_sat(src))));
        }
    }
}
