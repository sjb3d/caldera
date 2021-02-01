#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "payload.glsl"
#include "normal_pack.glsl"
#include "sampler.glsl"

layout(set = 0, binding = 0, scalar) uniform PathTraceData {
    mat4x3 world_from_camera;
    vec2 fov_size_at_unit_z;
    mat4x3 world_from_light;
    vec2 light_size_ws;
    vec3 light_emission;
} g_data;
layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, r16ui) uniform restrict readonly uimage2D g_samples;
layout(set = 0, binding = 3, r32f) uniform restrict image2D g_result_r;
layout(set = 0, binding = 4, r32f) uniform restrict image2D g_result_g;
layout(set = 0, binding = 5, r32f) uniform restrict image2D g_result_b;

#define SEQUENCE_COUNT        256

vec2 rand_u01(uvec2 pixel_coord, uint ray_index, uint sample_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((ray_index << 20) | (pixel_coord.y << 10) | pixel_coord.x);
    const ivec2 sample_coord = ivec2(sample_index, seq_hash & (SEQUENCE_COUNT - 1));
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

EXTEND_PAYLOAD_READ(g_extend);
OCCLUSION_PAYLOAD_READ(g_occlusion);

void main()
{
    const uvec2 pixel_coord = gl_LaunchIDEXT.xy;
    const uint sample_index = 0;

    vec3 prev_position;
    vec3 prev_normal;
    vec3 prev_in_dir;
    float prev_pdf_in;
    vec3 alpha;

    // sample the camera
    {
        const vec2 pixel_rand_u01 = rand_u01(pixel_coord, 0, sample_index);
        const vec2 fov_uv = (vec2(pixel_coord) + pixel_rand_u01)/vec2(gl_LaunchSizeEXT);
        const vec3 ray_dir_ls = normalize(vec3(g_data.fov_size_at_unit_z*(.5f - fov_uv), 1.f));
        const vec3 ray_dir = g_data.world_from_camera * vec4(ray_dir_ls, 0.f);
        
        const float fov_area_at_unit_z = mul_elements(g_data.fov_size_at_unit_z);
        const float cos_theta = ray_dir_ls.z;
        const float cos_theta2 = cos_theta*cos_theta;
        const float cos_theta4 = cos_theta2*cos_theta2;
        const float pdf_in = 1.f/(fov_area_at_unit_z*cos_theta4);

        const vec3 importance = vec3(1.f/fov_area_at_unit_z);
        const float area_pdf = 1.f;

        prev_position = g_data.world_from_camera[3];
        prev_normal = g_data.world_from_camera[2];
        prev_in_dir = ray_dir;
        prev_pdf_in = pdf_in;
        alpha = importance/area_pdf;
    }

    // trace a path from the camera
    vec3 result_sum = vec3(0.f);
    uint segment_index = 0;
    const uint max_segment_count = 3;
    for (;;) {
        // extend the path using the sampled (incoming) direction
        traceRayEXT(
            g_accel,
            gl_RayFlagsNoneEXT,
            0xff,
            EXTEND_HIT_SHADER_OFFSET,
            HIT_SHADER_COUNT_PER_INSTANCE,
            EXTEND_MISS_SHADER_OFFSET,
            prev_position,
            0.f,
            prev_in_dir,
            FLT_INF,
            EXTEND_PAYLOAD_INDEX);
        ++segment_index;

        // TODO: handle ray hitting a light source
        if ((g_extend.flags_packed & EXTEND_FLAGS_HAS_LIGHT_BIT) != 0) {

        }

        // end the ray if we didn't hit any surface
        if ((g_extend.flags_packed & EXTEND_FLAGS_HAS_SURFACE_BIT) == 0) {
            break;
        }
        if (segment_index == max_segment_count) {
            break;
        }

        // unpack the BRDF
        const vec2 reflection_and_emission_p0 = unpackHalf2x16(g_extend.reflectance_and_emission.x);
        const vec2 reflection_and_emission_p1 = unpackHalf2x16(g_extend.reflectance_and_emission.y);
        const vec3 hit_reflectance = vec3(reflection_and_emission_p0, reflection_and_emission_p1)/PI;

        const vec3 hit_position = g_extend.position;
        const mat3 hit_basis = basis_from_z_axis(normalize(vec_from_oct32(g_extend.normal_oct32)));
        const vec3 hit_normal = hit_basis[2];

        const vec3 out_dir_ls = normalize(-prev_in_dir * hit_basis);

        const int log2_epsilon_factor = -18;
        const float hit_epsilon = ldexp(1.f, int(g_extend.flags_packed & EXTEND_FLAGS_MAX_EXP_MASK) - 128 + log2_epsilon_factor);
        const vec3 adjusted_hit_pos = hit_position + hit_normal*hit_epsilon;

        // sample a light source
        {
            const vec2 light_rand_u01 = rand_u01(pixel_coord, 2*segment_index - 1, sample_index);
            const vec3 light_pos = g_data.world_from_light * vec4((light_rand_u01 - .5f) * g_data.light_size_ws, 0.f, 1.f);
            const vec3 light_normal = g_data.world_from_light[2];

            const float light_epsilon = ldexp(max_element(abs(g_data.world_from_light[3])) + max_element(abs(g_data.light_size_ws)), log2_epsilon_factor);
            const vec3 adjusted_light_pos = light_pos + light_normal*light_epsilon;

            const vec3 to_light_vec = adjusted_light_pos - adjusted_hit_pos;
            const float to_light_distance_sq = dot(to_light_vec, to_light_vec);
            const float to_light_distance = sqrt(to_light_distance_sq);
            const vec3 to_light_dir = to_light_vec / to_light_distance;

            const vec3 light_f = (dot(-to_light_dir, light_normal) > 0.f) ? g_data.light_emission : vec3(0.f);
            const vec3 light_alpha_f = light_f*mul_elements(g_data.light_size_ws); // emission/area_pdf

            const vec3 hit_f = (dot(to_light_dir, hit_normal) > 0.f) ? hit_reflectance : vec3(0.f);

            const float geometry_term
                = abs(dot(to_light_dir, hit_normal))
                * abs(dot(-to_light_dir, light_normal))
                / to_light_distance_sq;

            const vec3 unweighted_sample = alpha * hit_f * geometry_term * light_alpha_f;
            if (any(greaterThan(unweighted_sample, vec3(0.f)))) {
                const float distance_epsilon_factor = 1.f - ldexp(1.f, log2_epsilon_factor);        
                traceRayEXT(
                    g_accel,
                    gl_RayFlagsNoneEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                    0xff,
                    OCCLUSION_HIT_SHADER_OFFSET,
                    HIT_SHADER_COUNT_PER_INSTANCE,
                    OCCLUSION_MISS_SHADER_OFFSET,
                    adjusted_hit_pos,
                    0.f,
                    to_light_dir,
                    to_light_distance*distance_epsilon_factor,
                    OCCLUSION_PAYLOAD_INDEX);

                if (g_occlusion.is_occluded == 0) {
                    result_sum += unweighted_sample;
                }
            }
        }

        // TODO: RR once indirect

        // sample BSDF
        {
            const vec2 bsdf_rand_u01 = rand_u01(pixel_coord, 2*segment_index, sample_index);

            const vec3 in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);
            const float pdf_in = 1.f/PI;

            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = adjusted_hit_pos;
            prev_normal = hit_normal;
            prev_in_dir = in_dir;
            prev_pdf_in = pdf_in;
            alpha *= hit_reflectance/pdf_in;
        }
    }

    imageStore(g_result_r, ivec2(pixel_coord), vec4(result_sum.x, 0, 0, 0));
    imageStore(g_result_g, ivec2(pixel_coord), vec4(result_sum.y, 0, 0, 0));
    imageStore(g_result_b, ivec2(pixel_coord), vec4(result_sum.z, 0, 0, 0));
}
