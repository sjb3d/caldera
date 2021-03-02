#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"
#include "spectrum.glsl"

#define GROUP_X             8
#define GROUP_Y             8
#define GROUP_THREAD_COUNT  (GROUP_X*GROUP_Y)

layout(local_size_x = GROUP_X, local_size_y = GROUP_Y) in;

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
layout(set = 0, binding = 4, r32f) uniform restrict readonly image2D g_input[3];
layout(set = 0, binding = 5, rgba32f) uniform restrict image2D g_result;

#include "sequence.glsl"

vec4 rand_u01(uvec2 pixel_coord, uint seq_index)
{
    return rand_u01(pixel_coord, seq_index, g_filter.sample_index, g_filter.sequence_type);
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

#define TILE_X              (GROUP_X + 4)
#define TILE_Y              (GROUP_Y + 4)
#define TILE_PIXEL_COUNT    (TILE_X*TILE_Y)

shared float s_tile_r[TILE_PIXEL_COUNT];
shared float s_tile_g[TILE_PIXEL_COUNT];
shared float s_tile_b[TILE_PIXEL_COUNT];

vec3 load_input(uvec2 tile_coord_base, uvec2 load_coord)
{
    const uint shared_index = (load_coord.y - tile_coord_base.y)*TILE_X + (load_coord.x - tile_coord_base.x);
    vec3 value;
    value.x = s_tile_r[shared_index];
    value.y = s_tile_g[shared_index];
    value.z = s_tile_b[shared_index];
    return value;
}

void main()
{
    // all threads help to load a tile first
    const uvec2 tile_coord_base = gl_WorkGroupID.xy*uvec2(GROUP_X, GROUP_Y) - 2; 
    for (uint raster_index = gl_LocalInvocationIndex; raster_index < TILE_PIXEL_COUNT; raster_index += GROUP_THREAD_COUNT) {
        const uint local_y = raster_index/TILE_X;
        const uint local_x = raster_index - TILE_X*local_y;
        const uvec2 load_coord = tile_coord_base + uvec2(local_x, local_y);
        vec3 value = vec3(0.f);
        if (all(lessThan(load_coord, g_filter.image_size))) {
            vec3 samples;
            samples.x = imageLoad(g_input[0], ivec2(load_coord)).x;
            samples.y = imageLoad(g_input[1], ivec2(load_coord)).x;
            samples.z = imageLoad(g_input[2], ivec2(load_coord)).x;

            const vec4 pixel_rand_u01 = rand_u01(load_coord, 0);
            const float hero_wavelength = mix(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, pixel_rand_u01.x);
            const HERO_VEC wavelengths = expand_wavelengths(hero_wavelength);
            value
                = samples.x*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.x)).xyz
                + samples.y*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.y)).xyz
                + samples.z*texture(g_xyz_from_wavelength, unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelengths.z)).xyz
                ;
        }
        s_tile_r[raster_index] = value.x;
        s_tile_g[raster_index] = value.y;
        s_tile_b[raster_index] = value.z;
    }
    barrier();
    
    // now threads are allowed to exit
    const uvec2 pixel_coord = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_coord, g_filter.image_size))) {
        return;
    }

    vec4 result = vec4(0.f);

    switch (g_filter.filter_type) {
        default:
        case FILTER_TYPE_BOX: {
            result.xyz += load_input(tile_coord_base, pixel_coord);
            result.w += 1.f;
        } break;

        case FILTER_TYPE_GAUSSIAN: {
            for (int y = -2; y <= 2; ++y)
            for (int x = -2; x <= 2; ++x) {
                uvec2 load_coord = uvec2(ivec2(pixel_coord) + ivec2(x, y));
                if (any(greaterThanEqual(load_coord, g_filter.image_size))) {
                    continue;
                }

                const vec4 pixel_rand_u01 = rand_u01(load_coord, 1);
                const vec2 filter_coord = vec2(x, y) + pixel_rand_u01.xy - .5f;

                const float sigma2 = 2.2f;
                const float half_extent = 2.f;
                const float w = exp(-sigma2*dot(filter_coord, filter_coord)) - exp(-sigma2*half_extent*half_extent);
                if (w > 0.f) {
                    result.xyz += w*load_input(tile_coord_base, load_coord);
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

                const vec4 pixel_rand_u01 = rand_u01(load_coord, 1);
                const vec2 filter_coord = vec2(x, y) + pixel_rand_u01.xy - .5f;

                const float w = mitchell(filter_coord.x) * mitchell(filter_coord.y);
                if (w != 0.f) {
                    result.xyz += w*load_input(tile_coord_base, load_coord);
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
