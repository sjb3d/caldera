#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "occlusion_common.glsl"
#include "normal_pack.glsl"
#include "sampler.glsl"
#include "color_space.glsl"
#include "light_common.glsl"

layout(set = 0, binding = 0, scalar) uniform PathTraceUniforms {
    mat4x3 world_from_camera;
    vec2 fov_size_at_unit_z;
    uint sample_index;
    uint max_segment_count;
    uint render_color_space;
    uint use_max_roughness;
} g_path_trace;

LIGHT_UNIFORM_DATA(g_light);

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

layout(set = 0, binding = 2) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 3, r16ui) uniform restrict readonly uimage2D g_samples;
layout(set = 0, binding = 4, r32f) uniform restrict image2D g_result_r;
layout(set = 0, binding = 5, r32f) uniform restrict image2D g_result_g;
layout(set = 0, binding = 6, r32f) uniform restrict image2D g_result_b;

#define LOG2_SEQUENCE_COUNT     8

#define LOG2_EPSILON_FACTOR     (-18)

vec3 sample_from_rec709(vec3 c)
{
    switch (g_path_trace.render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return c;
        case RENDER_COLOR_SPACE_ACESCG: return acescg_from_rec709(c);
    }
}

vec2 rand_u01(uint seq_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((seq_index << 24) ^ (gl_LaunchIDEXT.y << 12) ^ gl_LaunchIDEXT.x);
    const ivec2 sample_coord = ivec2(g_path_trace.sample_index, seq_hash >> (32 - LOG2_SEQUENCE_COUNT));
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

EXTEND_PAYLOAD(g_extend);
OCCLUSION_PAYLOAD(g_occlusion);
LIGHT_EVAL_DATA(g_light_eval);
LIGHT_SAMPLE_DATA(g_light_sample);

void sample_single_light(
    uint light_index,
    vec3 target_position,
    vec3 target_normal,
    vec2 rand_u01,
    out vec3 light_position,
    out vec3 light_normal,
    out vec3 light_emission,
    out float light_solid_angle_pdf,
    out float light_epsilon)
{
    g_light_sample.position = target_position;
    g_light_sample.normal = target_normal;
    g_light_sample.emission = vec3(rand_u01, 0.f);
    executeCallableEXT(LIGHT_SAMPLE_SHADER_INDEX(light_index), LIGHT_SAMPLE_CALLABLE_INDEX);
    light_position = g_light_sample.position;
    light_normal = g_light_sample.normal;
    light_emission = sample_from_rec709(g_light_sample.emission);
    light_solid_angle_pdf = g_light_sample.solid_angle_pdf;
    light_epsilon = ldexp(g_light_sample.unit_value, LOG2_EPSILON_FACTOR);    
}

void sample_all_lights(
    vec3 target_position,
    vec3 target_normal,
    vec2 rand_u01,
    out vec3 light_position,
    out vec3 light_normal,
    out vec3 light_emission,
    out float light_solid_angle_pdf,
    out float light_epsilon)
{
    // pick a light source
    const float sampled_light_count_flt = float(g_light.sampled_light_count);
    const uint light_index = min(uint(rand_u01.x*sampled_light_count_flt), g_light.sampled_light_count - 1);
    rand_u01.x = fract(rand_u01.x*sampled_light_count_flt);
    const float selection_pdf = 1.f/sampled_light_count_flt;

    // sample this light
    sample_single_light(
        light_index,
        target_position,
        target_normal,
        rand_u01,
        light_position,
        light_normal,
        light_emission,
        light_solid_angle_pdf,
        light_epsilon);

    // adjust pdf for selection chance
    light_solid_angle_pdf *= selection_pdf;
}

void evaluate_single_light(
    uint light_index,
    vec3 prev_position,
    vec3 light_position,
    out vec3 light_emission,
    out float light_solid_angle_pdf)
{
    g_light_eval.position = light_position;
    g_light_eval.emission = prev_position;
    executeCallableEXT(LIGHT_EVAL_SHADER_INDEX(light_index), LIGHT_EVAL_CALLABLE_INDEX);
    light_emission = sample_from_rec709(g_light_eval.emission);
    light_solid_angle_pdf = g_light_eval.solid_angle_pdf;
}

float get_light_selection_pdf()
{
    const float sampled_light_count_flt = float(g_light.sampled_light_count);
    return 1.f/sampled_light_count_flt;
}

void main()
{
    vec3 prev_position;
    vec3 prev_normal;
    bool prev_is_delta;
    bool have_seen_non_delta;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_in_proj_solid_angle_pdf;
    vec3 alpha;

    // sample the camera
    {
        const vec2 pixel_rand_u01 = rand_u01(0);
        const vec2 fov_uv = (vec2(gl_LaunchIDEXT.xy) + pixel_rand_u01)/vec2(gl_LaunchSizeEXT);
        const vec3 ray_dir_ls = normalize(vec3(g_path_trace.fov_size_at_unit_z*(.5f - fov_uv), 1.f));
        const vec3 ray_dir = g_path_trace.world_from_camera * vec4(ray_dir_ls, 0.f);
        
        const float fov_area_at_unit_z = mul_elements(g_path_trace.fov_size_at_unit_z);
        const float cos_theta = ray_dir_ls.z;
        const float cos_theta2 = cos_theta*cos_theta;
        const float cos_theta4 = cos_theta2*cos_theta2;
        const float proj_solid_angle_pdf = 1.f/(fov_area_at_unit_z*cos_theta4);

        const vec3 importance = vec3(1.f/fov_area_at_unit_z);
        const float sensor_area_pdf = 1.f;

        prev_position = g_path_trace.world_from_camera[3];
        prev_normal = g_path_trace.world_from_camera[2];
        prev_is_delta = false;
        have_seen_non_delta = false;
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_in_proj_solid_angle_pdf = proj_solid_angle_pdf;
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
            const vec3 light_position = g_extend.position;
            vec3 light_emission;
            float light_solid_angle_pdf;
            evaluate_single_light(
                get_light_index(g_extend.hit),
                prev_position,
                light_position,
                light_emission,
                light_solid_angle_pdf);

            // take into account how it could have been sampled
            const bool light_can_be_sampled = has_surface(g_extend.hit);
            if (light_can_be_sampled) {
                light_solid_angle_pdf *= get_light_selection_pdf();
            }

            // compute the sample from this hit
            const vec3 unweighted_sample = alpha * light_emission;

            // apply MIS weight
            const bool can_be_sampled = (segment_index > 1) && light_can_be_sampled && !prev_is_delta;
            const float prev_in_cos_theta = abs(dot(prev_in_dir, prev_normal));
            const float prev_in_solid_angle_pdf = prev_in_cos_theta*prev_in_proj_solid_angle_pdf;
            const float other_ratio = can_be_sampled ? (light_solid_angle_pdf / prev_in_solid_angle_pdf) : 0.f;
            const float weight = 1.f/(1.f + other_ratio);
            const vec3 result = weight*unweighted_sample;

            result_sum += result;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(g_extend.hit)) {
            break;
        }
        if (segment_index >= g_path_trace.max_segment_count) {
            break;
        }

        // rewrite the BRDF to ensure roughness never reduces when extending an eye path
        if (g_path_trace.use_max_roughness != 0)
        if (have_seen_non_delta && get_bsdf_type(g_extend.hit) == BSDF_TYPE_MIRROR) {
            const vec3 mirror_reflectance = get_reflectance(g_extend.hit);
            const bool is_emissive = has_light(g_extend.hit);
            const uint light_index = get_light_index(g_extend.hit);
            const int max_exponent = get_max_exponent(g_extend.hit);
            g_extend.hit = create_hit_data(
                BSDF_TYPE_DIFFUSE,
                mirror_reflectance/PI,
                is_emissive,
                light_index,
                max_exponent);
        }

        // unpack the BRDF
        const vec3 hit_reflectance = sample_from_rec709(get_reflectance(g_extend.hit));

        const vec3 hit_position = g_extend.position;
        const mat3 hit_basis = basis_from_z_axis(normalize(vec_from_oct32(g_extend.normal_oct32)));
        const vec3 hit_normal = hit_basis[2];
        const bool hit_is_delta = (get_bsdf_type(g_extend.hit) == BSDF_TYPE_MIRROR);

        const vec3 out_dir_ls = normalize(-prev_in_dir * hit_basis);

        const float hit_epsilon = ldexp(1.f, get_max_exponent(g_extend.hit) + LOG2_EPSILON_FACTOR);
        const vec3 adjusted_hit_position = hit_position + hit_normal*hit_epsilon;

        // sample a light source
        if (g_light.sampled_light_count != 0 && !hit_is_delta) {
            // sample from all light sources
            vec3 light_position;
            vec3 light_normal;
            vec3 light_emission;
            float light_solid_angle_pdf;
            float light_epsilon;
            sample_all_lights(
                hit_position,
                hit_normal,
                rand_u01(2*segment_index - 1),
                light_position,
                light_normal,
                light_emission,
                light_solid_angle_pdf,
                light_epsilon);

            // evaluate the BRDF
            const vec3 light_from_hit_dir = normalize(light_position - hit_position);
            const float cos_theta = dot(light_from_hit_dir, hit_normal);
            const vec3 hit_f = (cos_theta > 0.f) ? hit_reflectance : vec3(0.f);
            const float hit_proj_solid_angle_pdf = get_hemisphere_cosine_weighted_proj_pdf();
            
            // compute the sample assuming the ray is not occluded
            const float rcp_light_proj_solid_angle_pdf = abs(cos_theta)/light_solid_angle_pdf;
            const vec3 unweighted_sample = alpha * hit_f * rcp_light_proj_solid_angle_pdf * light_emission;

            // apply MIS weight
            const bool light_can_be_hit = true;
            const float other_ratio = light_can_be_hit ? (hit_proj_solid_angle_pdf * rcp_light_proj_solid_angle_pdf) : 0.f;
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
            vec3 in_dir_ls;
            float in_proj_solid_angle_pdf;
            switch (get_bsdf_type(g_extend.hit)) {
                default:
                case BSDF_TYPE_DIFFUSE: {
                    const vec2 bsdf_rand_u01 = rand_u01(2*segment_index);
                    in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);
                    in_proj_solid_angle_pdf = get_hemisphere_cosine_weighted_proj_pdf();
                } break;

                case BSDF_TYPE_MIRROR: {
                    in_dir_ls = vec3(-out_dir_ls.xy, out_dir_ls.z);
                    in_proj_solid_angle_pdf = 1.f;
                } break;
            }
            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = hit_position;
            prev_normal = hit_normal;
            prev_is_delta = hit_is_delta;
            if (!hit_is_delta) {
                have_seen_non_delta = true;
            }
            prev_epsilon = hit_epsilon;
            prev_in_dir = in_dir;
            prev_in_proj_solid_angle_pdf = in_proj_solid_angle_pdf;
            alpha *= hit_reflectance/in_proj_solid_angle_pdf;
        }
    }

    if (g_path_trace.sample_index != 0) {
        result_sum.r += imageLoad(g_result_r, ivec2(gl_LaunchIDEXT.xy)).x;
        result_sum.g += imageLoad(g_result_g, ivec2(gl_LaunchIDEXT.xy)).x;
        result_sum.b += imageLoad(g_result_b, ivec2(gl_LaunchIDEXT.xy)).x;
    }
    imageStore(g_result_r, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.x, 0, 0, 0));
    imageStore(g_result_g, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.y, 0, 0, 0));
    imageStore(g_result_b, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.z, 0, 0, 0));
}
