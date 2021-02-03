#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "payload.glsl"
#include "normal_pack.glsl"
#include "sampler.glsl"
#include "color_space.glsl"

layout(set = 0, binding = 0, scalar) uniform PathTraceData {
    mat4x3 world_from_camera;
    vec2 fov_size_at_unit_z;
    mat4x3 world_from_light;
    vec2 light_size_ws;
    vec3 light_emission;
    uint sample_index;
    uint max_segment_count;
    uint render_color_space;
} g_data;

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, r16ui) uniform restrict readonly uimage2D g_samples;
layout(set = 0, binding = 3, r32f) uniform restrict image2D g_result_r;
layout(set = 0, binding = 4, r32f) uniform restrict image2D g_result_g;
layout(set = 0, binding = 5, r32f) uniform restrict image2D g_result_b;

#define SEQUENCE_COUNT          256

#define LOG2_EPSILON_FACTOR     (-18)

vec3 sample_from_rec709(vec3 c)
{
    switch (g_data.render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return c;
        case RENDER_COLOR_SPACE_ACESCG: return acescg_from_rec709(c);
    }
}

vec2 rand_u01(uint ray_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((ray_index << 20) | (gl_LaunchIDEXT.y << 10) | gl_LaunchIDEXT.x);
    const ivec2 sample_coord = ivec2(g_data.sample_index, seq_hash & (SEQUENCE_COUNT - 1));
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

float sample_lights(
    uint segment_index,
    vec3 target_position,
    vec3 target_normal,
    out vec3 light_position,
    out vec3 light_normal,
    out float light_epsilon,
    out vec3 light_emission,
    out float light_area_pdf)
{
    const vec2 light_rand_u01 = rand_u01(2*segment_index - 1);

    light_position = g_data.world_from_light * vec4((light_rand_u01 - .5f) * g_data.light_size_ws, 0.f, 1.f);
    light_normal = g_data.world_from_light[2];
    light_epsilon = ldexp(max_element(abs(g_data.world_from_light[3])) + max_element(abs(g_data.light_size_ws)), LOG2_EPSILON_FACTOR);

    const vec3 target_from_light = target_position - light_position;
    const float light_facing_term = dot(target_from_light, light_normal);
    light_emission = (light_facing_term > 0.f) ? sample_from_rec709(g_data.light_emission) : vec3(0.f);
    light_area_pdf = 1.f/mul_elements(g_data.light_size_ws);

    const float target_facing_term = dot(target_from_light, target_normal);
    const float distance_sq = dot(target_from_light, target_from_light);
    return abs(light_facing_term * target_facing_term) / (distance_sq * distance_sq);
}

float evaluate_hit_light(
    vec3 prev_position,
    vec3 prev_normal,
    vec3 light_position,
    vec3 light_normal,
    out vec3 light_emission,
    out float light_area_pdf)
{
    const vec3 prev_from_light = prev_position - light_position;
    const float light_facing_term = dot(prev_from_light, light_normal);
    light_emission = (light_facing_term > 0.f) ? sample_from_rec709(g_data.light_emission) : vec3(0.f);
    light_area_pdf = 1.f/mul_elements(g_data.light_size_ws);

    const float prev_facing_term = dot(prev_from_light, prev_normal);
    const float distance_sq = dot(prev_from_light, prev_from_light);
    return abs(light_facing_term * prev_facing_term) / (distance_sq * distance_sq);
}


EXTEND_PAYLOAD_READ(g_extend);
OCCLUSION_PAYLOAD_READ(g_occlusion);

void main()
{
    vec3 prev_position;
    vec3 prev_normal;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_psa_pdf_in;
    vec3 alpha;

    // sample the camera
    {
        const vec2 pixel_rand_u01 = rand_u01(0);
        const vec2 fov_uv = (vec2(gl_LaunchIDEXT.xy) + pixel_rand_u01)/vec2(gl_LaunchSizeEXT);
        const vec3 ray_dir_ls = normalize(vec3(g_data.fov_size_at_unit_z*(.5f - fov_uv), 1.f));
        const vec3 ray_dir = g_data.world_from_camera * vec4(ray_dir_ls, 0.f);
        
        const float fov_area_at_unit_z = mul_elements(g_data.fov_size_at_unit_z);
        const float cos_theta = ray_dir_ls.z;
        const float cos_theta2 = cos_theta*cos_theta;
        const float cos_theta4 = cos_theta2*cos_theta2;
        const float psa_pdf_in = 1.f/(fov_area_at_unit_z*cos_theta4);

        const vec3 importance = vec3(1.f/fov_area_at_unit_z);
        const float sensor_area_pdf = 1.f;

        prev_position = g_data.world_from_camera[3];
        prev_normal = g_data.world_from_camera[2];
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_psa_pdf_in = psa_pdf_in;
        alpha = importance/sensor_area_pdf;
    }

    // trace a path from the camera
    vec3 result_sum = vec3(0.f);
    uint segment_index = 0;
    for (;;) {
        // extend the path using the sampled (incoming) direction
        traceRayEXT(
            g_accel,
            gl_RayFlagsNoneEXT,
            0xff,
            EXTEND_HIT_SHADER_OFFSET,
            HIT_SHADER_COUNT_PER_INSTANCE,
            EXTEND_MISS_SHADER_OFFSET,
            prev_position + prev_epsilon*prev_normal,
            0.f,
            prev_in_dir,
            FLT_INF,
            EXTEND_PAYLOAD_INDEX);
        ++segment_index;

        // handle ray hitting a light source
        if (has_light(g_extend.hit)) {
            // evaluate the light here
            vec3 light_emission;
            float light_area_pdf;
            const float geometry_term = evaluate_hit_light(
                prev_position,
                prev_normal,
                g_extend.position,
                g_data.world_from_light[2],
                light_emission,
                light_area_pdf);

            const vec3 unweighted_sample = alpha * light_emission;

            // apply MIS weight
            const bool can_be_sampled = (segment_index > 1);
            const float other_ratio = can_be_sampled ? (light_area_pdf / (geometry_term * prev_psa_pdf_in)) : 0.f;
            const float weight = 1.f/(1.f + other_ratio);
            const vec3 result = weight*unweighted_sample;

            result_sum += result;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(g_extend.hit)) {
            break;
        }
        if (segment_index >= g_data.max_segment_count) {
            break;
        }

        // unpack the BRDF
        const vec3 hit_reflectance = sample_from_rec709(get_reflectance(g_extend.hit))/PI;

        const vec3 hit_position = g_extend.position;
        const mat3 hit_basis = basis_from_z_axis(normalize(vec_from_oct32(g_extend.normal_oct32)));
        const vec3 hit_normal = hit_basis[2];

        const vec3 out_dir_ls = normalize(-prev_in_dir * hit_basis);

        const float hit_epsilon = ldexp(1.f, get_max_exponent(g_extend.hit) + LOG2_EPSILON_FACTOR);
        const vec3 adjusted_hit_position = hit_position + hit_normal*hit_epsilon;

        // sample a light source
        {
            // sample from all light sources
            vec3 light_position;
            vec3 light_normal;
            float light_epsilon;
            vec3 light_emission;
            float light_area_pdf;
            const float geometry_term = sample_lights(
                segment_index,
                hit_position,
                hit_normal,
                light_position,
                light_normal,
                light_epsilon,
                light_emission,
                light_area_pdf);

            // evaluate the BRDF
            const vec3 hit_f = (dot(light_position - hit_position, hit_normal) > 0.f) ? hit_reflectance : vec3(0.f);
            const float hit_psa_pdf_in = get_hemisphere_cosine_weighted_psa_pdf();
            
            // compute the sample assuming the ray is not occluded
            const vec3 light_alpha_f = light_emission/light_area_pdf;
            const vec3 unweighted_sample = alpha * hit_f * geometry_term * light_alpha_f;

            // apply MIS weight
            const bool light_can_be_hit = true;
            const float other_ratio = light_can_be_hit ? (geometry_term * hit_psa_pdf_in / light_area_pdf) : 0.f;
            const float weight = 1.f/(1.f + other_ratio);
            const vec3 result = weight*unweighted_sample;

            // trace an occlusion ray if necessary
            if (any(greaterThan(result, vec3(0.f)))) {
                const vec3 adjusted_light_position = light_position + light_normal*light_epsilon;

                const vec3 ray_origin = adjusted_hit_position;
                const vec3 ray_vec = adjusted_light_position - ray_origin;
                const float ray_distance = length(ray_vec);
                const vec3 ray_dir = ray_vec/ray_distance;

                const float distance_epsilon_factor = 1.f - ldexp(1.f, LOG2_EPSILON_FACTOR);
                traceRayEXT(
                    g_accel,
                    gl_RayFlagsNoneEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                    0xff,
                    OCCLUSION_HIT_SHADER_OFFSET,
                    HIT_SHADER_COUNT_PER_INSTANCE,
                    OCCLUSION_MISS_SHADER_OFFSET,
                    ray_origin,
                    0.f,
                    ray_dir,
                    ray_distance*distance_epsilon_factor,
                    OCCLUSION_PAYLOAD_INDEX);

                if (g_occlusion.is_occluded == 0) {
                    result_sum += result;
                }
            }
        }

        // TODO: RR once indirect

        // sample BSDF
        {
            const vec2 bsdf_rand_u01 = rand_u01(2*segment_index);

            const vec3 in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);
            const float psa_pdf_in = get_hemisphere_cosine_weighted_psa_pdf();

            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = hit_position;
            prev_normal = hit_normal;
            prev_epsilon = hit_epsilon;
            prev_in_dir = in_dir;
            prev_psa_pdf_in = psa_pdf_in;
            alpha *= hit_reflectance/psa_pdf_in;
        }
    }

    if (g_data.sample_index != 0) {
        result_sum.r += imageLoad(g_result_r, ivec2(gl_LaunchIDEXT.xy)).x;
        result_sum.g += imageLoad(g_result_g, ivec2(gl_LaunchIDEXT.xy)).x;
        result_sum.b += imageLoad(g_result_b, ivec2(gl_LaunchIDEXT.xy)).x;
    }
    imageStore(g_result_r, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.x, 0, 0, 0));
    imageStore(g_result_g, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.y, 0, 0, 0));
    imageStore(g_result_b, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.z, 0, 0, 0));
}
