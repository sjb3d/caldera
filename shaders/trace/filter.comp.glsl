#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"
#include "spectrum.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

#define FILTER_TYPE_BOX         0x0
#define FILTER_TYPE_GAUSSIAN    0x1
#define FILTER_TYPE_MITCHELL    0x2

layout(set = 0, binding = 0, scalar) uniform FilterData {
    uvec2 image_size;
    uint sequence_type;
    uint sample_index;
    uint filter_type;
} g_filter;

layout(set = 0, binding = 1, rg32f) uniform restrict readonly image2D g_pmj_samples;
layout(set = 0, binding = 2, rgba32ui) uniform restrict readonly uimage2D g_sobol_samples;
layout(set = 0, binding = 3) uniform sampler1D g_xyz_from_wavelength;
layout(set = 0, binding = 4, rgba32f) uniform restrict readonly image2D g_input;
layout(set = 0, binding = 5, rgba32f) uniform restrict image2D g_result;

#include "sequence.glsl"

vec4 rand_u01(uvec2 pixel_coord)
{
    return rand_u01(pixel_coord, 0, g_filter.sample_index, g_filter.sequence_type);
}

float mitchell(float x)
{
    const float B = 1.f/3.f;
    const float C = 1.f/3.f;
    
    x = abs(x);
    const float x2 = x*x;
    const float x3 = x2*x; 

    float y = 0.f;
    if (x < 1.f) {
        y = ((12.f - 9.f*B - 6.f*C)*x3 + (-18.f + 12.f*B + 6.f*C)*x2 + (6.f - 2.f*B))/6.f;
    } else if (x < 2.f) {
        y = ((-B - 6.f*C)*x3 + (6.f*B + 30.f*C)*x2 + (-12.f*B - 48.f*C)*x + (8.f*B + 24.f*C))/6.f;
    }
    return y;
}

vec3 load_input(uvec2 load_coord, float hero_wavelength)
{
    vec4 wavelengths;
    wavelengths.x = hero_wavelength;
    wavelengths.y = offset_wavelength(hero_wavelength, .25f);
    wavelengths.z = offset_wavelength(hero_wavelength, .5f);
    wavelengths.w = offset_wavelength(hero_wavelength, .75f);
    const vec4 samples = imageLoad(g_input, ivec2(load_coord));
    vec3 value
        = samples.x*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.x)).xyz
        + samples.y*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.y)).xyz
        + samples.z*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.z)).xyz
        + samples.w*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.w)).xyz;    
    return value;
}

void main()
{
    const uvec2 pixel_coord = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_coord, g_filter.image_size))) {
        return;
    }

    vec4 result = vec4(0.f);

    switch (g_filter.filter_type) {
        default:
        case FILTER_TYPE_BOX: {
            const vec4 pixel_rand_u01 = rand_u01(pixel_coord);
            const float hero_wavelength = mix(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, pixel_rand_u01.z);

            result.xyz += load_input(pixel_coord, hero_wavelength);
            result.w += 1.f;
        } break;

        case FILTER_TYPE_GAUSSIAN: {
            for (int y = -2; y <= 2; ++y)
            for (int x = -2; x <= 2; ++x) {
                uvec2 load_coord = uvec2(ivec2(pixel_coord) + ivec2(x, y));
                if (any(greaterThanEqual(load_coord, g_filter.image_size))) {
                    continue;
                }

                const vec4 pixel_rand_u01 = rand_u01(load_coord);
                const vec2 filter_coord = vec2(x, y) + pixel_rand_u01.xy - .5f;
                const float hero_wavelength = mix(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, pixel_rand_u01.z);

                const float sigma2 = 2.2f;
                const float half_extent = 2.f;
                const float w = exp(-sigma2*dot(filter_coord, filter_coord)) - exp(-sigma2*half_extent*half_extent);
                if (w > 0.f) {
                    result.xyz += w*load_input(load_coord, hero_wavelength);
                    result.w += w;
                }
            }
        } break;

        case FILTER_TYPE_MITCHELL: {
            for (int y = -2; y <= 2; ++y)
            for (int x = -2; x <= 2; ++x) {
                uvec2 load_coord = uvec2(ivec2(pixel_coord) + ivec2(x, y));
                if (any(greaterThanEqual(load_coord, g_filter.image_size))) {
                    continue;
                }

                const vec4 pixel_rand_u01 = rand_u01(load_coord);
                const vec2 filter_coord = vec2(x, y) + pixel_rand_u01.xy - .5f;
                const float hero_wavelength = mix(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, pixel_rand_u01.z);

                const float w = mitchell(filter_coord.x) * mitchell(filter_coord.y);
                if (w != 0.f) {
                    result.xyz += w*load_input(load_coord, hero_wavelength);
                    result.w += w;
                }
            }
        } break;
    }
    
    if (g_filter.sample_index != 0) {
        result += imageLoad(g_result, ivec2(pixel_coord));
    }
    imageStore(g_result, ivec2(pixel_coord), result);
}
