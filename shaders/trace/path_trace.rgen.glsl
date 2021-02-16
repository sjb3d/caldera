#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "occlusion_common.glsl"
#include "sampler.glsl"
#include "color_space.glsl"
#include "light_common.glsl"
#include "ggx.glsl"
#include "fresnel.glsl"
#include "rand_common.glsl"

#include "diffuse_bsdf.glsl"
#include "mirror_bsdf.glsl"
#include "smooth_dielectric_bsdf.glsl"
#include "smooth_plastic_bsdf.glsl"
#include "rough_plastic_bsdf.glsl"
#include "rough_conductor_bsdf.glsl"

#define PATH_TRACE_FLAG_ACCUMULATE_ROUGHNESS    0x01
#define PATH_TRACE_FLAG_ALLOW_LIGHT_SAMPLING    0x02
#define PATH_TRACE_FLAG_ALLOW_BSDF_SAMPLING     0x04

#define MIS_HEURISTIC_NONE      0
#define MIS_HEURISTIC_BALANCE   1
#define MIS_HEURISTIC_POWER2    2

layout(set = 0, binding = 0, scalar) uniform PathTraceUniforms {
    mat4x3 world_from_camera;
    vec2 fov_size_at_unit_z;
    uint sample_index;
    uint max_segment_count;
    uint render_color_space;
    uint mis_heuristic;
    uint flags;
} g_path_trace;

LIGHT_UNIFORM_DATA(g_light);

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

layout(set = 0, binding = 2) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 3, rg16ui) uniform restrict readonly uimage2D g_samples;
layout(set = 0, binding = 4, r32f) uniform restrict writeonly image2D g_result[3];

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
    const ivec2 sample_coord = rand_sample_coord(gl_LaunchIDEXT.xy, seq_index, g_path_trace.sample_index);
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

uint sample_uniform_discrete(uint count, inout float u01)
{
    const float index_flt = float(count)*u01;
    const uint index = min(uint(index_flt), count - 1);
    u01 = index_flt - float(index);
    return index;
}

float luminance(vec3 c)
{
    switch (g_path_trace.render_color_space) {
        default:
        case RENDER_COLOR_SPACE_REC709: return dot(c, vec3(0.2126729f, 0.7151522f, 0.0721750f));
        case RENDER_COLOR_SPACE_ACESCG: return dot(c, vec3(0.2722287f, 0.6740818f, 0.0536895f));
    }
}

EXTEND_PAYLOAD(g_extend);
OCCLUSION_PAYLOAD(g_occlusion);
LIGHT_EVAL_DATA(g_light_eval);
LIGHT_SAMPLE_DATA(g_light_sample);

void evaluate_bsdf(
    uint bsdf_type,
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vec3 f,
    out float solid_angle_pdf)
{
#define CALL(F) F(out_dir, in_dir, params, f, solid_angle_pdf)
    switch (bsdf_type) {
        case BSDF_TYPE_DIFFUSE:             CALL(diffuse_bsdf_eval); break;
        case BSDF_TYPE_MIRROR:              break;
        case BSDF_TYPE_SMOOTH_DIELECTRIC:   break;
        case BSDF_TYPE_SMOOTH_PLASTIC:      CALL(smooth_plastic_bsdf_eval); break;
        case BSDF_TYPE_ROUGH_PLASTIC:       CALL(rough_plastic_bsdf_eval); break;
        case BSDF_TYPE_ROUGH_CONDUCTOR:     CALL(rough_conductor_bsdf_eval); break;
    }
#undef CALL
}

void sample_bsdf(
    uint bsdf_type,
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    out float sampled_roughness)
{
#define CALL(F) F(out_dir, params, rand_u01, in_dir, estimator, solid_angle_pdf_or_negative, sampled_roughness)
    switch (bsdf_type) {
        case BSDF_TYPE_DIFFUSE:             CALL(diffuse_bsdf_sample); break;
        case BSDF_TYPE_MIRROR:              CALL(mirror_bsdf_sample); break;
        case BSDF_TYPE_SMOOTH_DIELECTRIC:   CALL(smooth_dielectric_bsdf_sample); break;
        case BSDF_TYPE_SMOOTH_PLASTIC:      CALL(smooth_plastic_bsdf_sample); break;
        case BSDF_TYPE_ROUGH_PLASTIC:       CALL(rough_plastic_bsdf_sample); break;
        case BSDF_TYPE_ROUGH_CONDUCTOR:     CALL(rough_conductor_bsdf_sample); break;
    }
#undef CALL
}

void sample_single_light(
    uint light_index,
    vec3 target_position,
    vec3 target_normal,
    vec2 rand_u01,
    out vec3 light_position_or_extdir,
    out vec3 light_normal,
    out vec3 light_emission,
    out float light_solid_angle_pdf,
    out bool light_is_external,
    out float light_epsilon)
{
    // c.f. LightSampleData
    g_light_sample.a = target_position;
    g_light_sample.b = target_normal;
    g_light_sample.c.xy = rand_u01;

    executeCallableEXT(LIGHT_SAMPLE_SHADER_INDEX(light_index), LIGHT_SAMPLE_CALLABLE_INDEX);

    // c.f. write_light_sample_outputs
    light_position_or_extdir = g_light_sample.a;
    light_normal = normalize(vec_from_oct32(floatBitsToUint(g_light_sample.c.x)));
    light_emission = sample_from_rec709(g_light_sample.b);
    light_solid_angle_pdf = abs(g_light_sample.c.y);
    light_is_external = sign_bit_set(g_light_sample.c.y);
    light_epsilon = ldexp(g_light_sample.c.z, LOG2_EPSILON_FACTOR);    
}

void sample_all_lights(
    vec3 target_position,
    vec3 target_normal,
    vec2 rand_u01,
    out vec3 light_position_or_extdir,
    out vec3 light_normal,
    out vec3 light_emission,
    out float light_solid_angle_pdf,
    out bool light_is_external,
    out float light_epsilon)
{
    // pick a light source
    const uint entry_index = sample_uniform_discrete(g_light.sampled_count, rand_u01.x);
    const LightAliasEntry entry = g_light.alias_table.entries[entry_index];
    uint light_index;
    if (split_random_variable(entry.split, rand_u01.y)) {
        light_index = entry.indices & 0xffffU;
    } else {
        light_index = entry.indices >> 16;
    }
    const float selection_pdf = g_light.probability_table.entries[light_index];

    // sample this light
    sample_single_light(
        light_index,
        target_position,
        target_normal,
        rand_u01,
        light_position_or_extdir,
        light_normal,
        light_emission,
        light_solid_angle_pdf,
        light_is_external,
        light_epsilon);

    // adjust pdf for selection chance
    light_solid_angle_pdf *= selection_pdf;
}

void evaluate_single_light(
    uint light_index,
    vec3 prev_position,
    vec3 light_position_or_extdir,
    out vec3 light_emission,
    out float light_solid_angle_pdf)
{
    // c.f. LightEvalData
    g_light_eval.a = light_position_or_extdir;
    g_light_eval.b = prev_position;

    executeCallableEXT(LIGHT_EVAL_SHADER_INDEX(light_index), LIGHT_EVAL_CALLABLE_INDEX);

    // c.f. write_light_eval_outputs
    light_emission = sample_from_rec709(g_light_eval.a);
    light_solid_angle_pdf = g_light_eval.b.x;
}

float get_light_selection_pdf(uint light_index)
{
    return g_light.probability_table.entries[light_index];
}

float mis_ratio(float ratio)
{
    switch (g_path_trace.mis_heuristic) {
        default:
        case MIS_HEURISTIC_NONE:    return 1.f;
        case MIS_HEURISTIC_BALANCE: return ratio;
        case MIS_HEURISTIC_POWER2:  return ratio*ratio;
    }
}

ExtendPayload trace_extend_ray(
    vec3 ray_origin,
    vec3 ray_dir)
{
    traceRayEXT(
        g_accel,
        gl_RayFlagsNoneEXT,
        0xff,
        EXTEND_HIT_SHADER_OFFSET,
        HIT_SHADER_COUNT_PER_INSTANCE,
        EXTEND_MISS_SHADER_OFFSET,
        ray_origin,
        0.f,
        ray_dir,
        FLT_INF,
        EXTEND_PAYLOAD_INDEX);
    return g_extend;
}

bool trace_occlusion_ray(
    vec3 ray_origin,
    vec3 ray_dir,
    float ray_distance)
{
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
        ray_distance,
        OCCLUSION_PAYLOAD_INDEX);
    return g_occlusion.is_occluded != 0;
}

void main()
{
    const bool allow_light_sampling = ((g_path_trace.flags & PATH_TRACE_FLAG_ALLOW_LIGHT_SAMPLING) != 0);
    const bool allow_bsdf_sampling = ((g_path_trace.flags & PATH_TRACE_FLAG_ALLOW_BSDF_SAMPLING) != 0);
    const bool accumulate_roughness = ((g_path_trace.flags & PATH_TRACE_FLAG_ACCUMULATE_ROUGHNESS) != 0);

    vec3 prev_position;
    Normal32 prev_geom_normal_packed;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_in_solid_angle_pdf_or_negative;
    vec3 prev_sample;
    float roughness_acc;

    // sample the camera
    {
        const vec2 pixel_rand_u01 = rand_u01(0);
        const vec2 fov_uv = (vec2(gl_LaunchIDEXT.xy) + pixel_rand_u01)/vec2(gl_LaunchSizeEXT);
        const vec3 ray_dir_ls = normalize(vec3(g_path_trace.fov_size_at_unit_z*(.5f - fov_uv), 1.f));
        const vec3 ray_dir = g_path_trace.world_from_camera * vec4(ray_dir_ls, 0.f);
        
        const float fov_area_at_unit_z = mul_elements(g_path_trace.fov_size_at_unit_z);
        const float cos_theta = ray_dir_ls.z;
        const float cos_theta2 = cos_theta*cos_theta;
        const float cos_theta3 = cos_theta2*cos_theta;
        const float solid_angle_pdf = 1.f/(fov_area_at_unit_z*cos_theta3);

        const vec3 importance = vec3(1.f/fov_area_at_unit_z);
        const float sensor_area_pdf = 1.f/fov_area_at_unit_z;

        prev_position = g_path_trace.world_from_camera[3];
        prev_geom_normal_packed = make_normal32(g_path_trace.world_from_camera[2]);
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_in_solid_angle_pdf_or_negative = solid_angle_pdf;
        prev_sample = importance/sensor_area_pdf;
        roughness_acc = 0.f;
    }

    // trace a path from the camera
    vec3 result_sum = vec3(0.f);
    uint segment_index = 0;
    for (;;) {
        // extend the path using the sampled (incoming) direction
        ExtendPayload hit = trace_extend_ray(
            prev_position + prev_epsilon*get_dir(prev_geom_normal_packed),
            prev_in_dir);
        ++segment_index;

        // adjust BSDF colour space of the hit
        hit.bsdf_params = replace_reflectance(
            hit.bsdf_params,
            sample_from_rec709(get_reflectance(hit.bsdf_params)));

        // reject implausible shading normals
        {
            const vec3 hit_shading_normal_vec = get_vec(hit.shading_normal);
            if (has_surface(hit.info) && dot(hit_shading_normal_vec, prev_in_dir) >= 0.f) {
                break;
            }
        }

        // handle ray hitting a light source
        uint light_index_begin = 0;
        uint light_index_end = 0;
        if (allow_bsdf_sampling || roughness_acc == 0.f) {
            if (has_light(hit.info)) {
                light_index_begin = get_light_index(hit.info);
                light_index_end = light_index_begin + 1;
            } else if (!has_surface(hit.info)) {
                light_index_begin = g_light.external_begin;
                light_index_end = g_light.external_end;
            }
        }
        for (uint light_index = light_index_begin; light_index != light_index_end; ++light_index) {
            // evaluate the light here
            const vec3 light_position_or_extdir = hit.position_or_extdir;
            vec3 light_emission;
            float light_solid_angle_pdf;
            evaluate_single_light(
                light_index,
                prev_position,
                light_position_or_extdir,
                light_emission,
                light_solid_angle_pdf);

            // take into account how it could have been sampled
            const bool light_can_be_sampled = allow_light_sampling && (light_index < g_light.sampled_count);
            if (light_can_be_sampled) {
                light_solid_angle_pdf *= get_light_selection_pdf(light_index);
            }

            // compute MIS weight
            const bool prev_is_delta = sign_bit_set(prev_in_solid_angle_pdf_or_negative);
            const bool can_be_sampled = (segment_index > 1) && light_can_be_sampled && !prev_is_delta;
            const float other_ratio = can_be_sampled ? mis_ratio(light_solid_angle_pdf / prev_in_solid_angle_pdf_or_negative) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // accumulate this sample
            result_sum += prev_sample * light_emission * mis_weight;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(hit.info)) {
            break;
        }
        if (segment_index >= g_path_trace.max_segment_count) {
            break;
        }

        // rewrite the BRDF to ensure roughness never reduces when extending an eye path
        if (accumulate_roughness) {
            const float orig_roughness = get_roughness(hit.bsdf_params);
            const float p = 4.f;
            roughness_acc = min(pow(pow(roughness_acc, p) + pow(orig_roughness, p), 1.f/p), 1.f);

            switch (get_bsdf_type(hit.info)) {
                case BSDF_TYPE_MIRROR:
                case BSDF_TYPE_ROUGH_CONDUCTOR:
                    if (roughness_acc != 0.f) {
                        const uint bsdf_type = (roughness_acc < 1.f) ? BSDF_TYPE_ROUGH_CONDUCTOR : BSDF_TYPE_DIFFUSE;
                        hit.info = replace_bsdf_type(hit.info, bsdf_type);
                        hit.bsdf_params = replace_roughness(hit.bsdf_params, roughness_acc);
                    }
                    break;
                case BSDF_TYPE_SMOOTH_DIELECTRIC:
                    // TODO: degrade to rough dielectric
                    break;
                case BSDF_TYPE_DIFFUSE:
                    // nothing to do
                    break;
                case BSDF_TYPE_ROUGH_PLASTIC:
                case BSDF_TYPE_SMOOTH_PLASTIC:
                    if (roughness_acc != 0.f) {
                        const uint bsdf_type = (roughness_acc < 1.f) ? BSDF_TYPE_ROUGH_PLASTIC : BSDF_TYPE_DIFFUSE;
                        hit.info = replace_bsdf_type(hit.info, bsdf_type);
                        hit.bsdf_params = replace_roughness(hit.bsdf_params, roughness_acc);
                    }
                    break;
            }
        }

        // unpack the BRDF
        const vec3 out_dir_ls = normalize(-prev_in_dir * basis_from_z_axis(get_dir(hit.shading_normal)));

        // sample a light source
        const bool hit_is_always_delta = bsdf_is_always_delta(get_bsdf_type(hit.info));
        if (g_light.sampled_count != 0 && allow_light_sampling && !hit_is_always_delta) {
            // sample from all light sources
            vec3 light_position_or_extdir;
            vec3 light_normal;
            vec3 light_emission;
            float light_solid_angle_pdf;
            bool light_is_external;
            float light_epsilon;
            sample_all_lights(
                hit.position_or_extdir,
                get_dir(hit.shading_normal),
                rand_u01(2*segment_index - 1),
                light_position_or_extdir,
                light_normal,
                light_emission,
                light_solid_angle_pdf,
                light_is_external,
                light_epsilon);

            vec3 in_vec_ws;
            if (light_is_external) {
                in_vec_ws = light_position_or_extdir;
            } else {
                in_vec_ws = light_position_or_extdir - hit.position_or_extdir;
            }
            const mat3 hit_basis = basis_from_z_axis(get_dir(hit.shading_normal));
            const vec3 in_dir_ls = normalize(in_vec_ws*hit_basis);

            // evaluate the BRDF
            const float in_cos_theta = in_dir_ls.z;
            const uint bsdf_type = get_bsdf_type(hit.info);
            vec3 hit_f;
            float hit_solid_angle_pdf;
            if (in_cos_theta > 0.f || bsdf_has_transmission(bsdf_type)) {
                evaluate_bsdf(
                    bsdf_type,
                    out_dir_ls,
                    in_dir_ls,
                    hit.bsdf_params,
                    hit_f,
                    hit_solid_angle_pdf);
            } else {
                hit_f = vec3(0.f);
                hit_solid_angle_pdf = 0.f;
            }
            
            // compute MIS weight
            const bool light_can_be_hit = allow_bsdf_sampling;
            const float other_ratio = light_can_be_hit ? mis_ratio(hit_solid_angle_pdf/light_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // compute the sample assuming the ray is not occluded
            const vec3 result = prev_sample * hit_f * (mis_weight*abs(in_cos_theta)/light_solid_angle_pdf) * light_emission;

            // trace an occlusion ray if necessary
            if (any(greaterThan(result, vec3(0.f)))) {
                const vec3 hit_geom_normal = get_dir(hit.geom_normal);
                const float hit_epsilon = get_epsilon(hit.info, LOG2_EPSILON_FACTOR);
                const vec3 adjusted_hit_position = hit.position_or_extdir + hit_geom_normal*hit_epsilon;

                vec3 ray_origin;
                vec3 ray_dir;
                float ray_distance;
                if (light_is_external) {
                    ray_origin = adjusted_hit_position;
                    ray_dir = light_position_or_extdir;
                    ray_distance = FLT_INF;
                } else {
                    const vec3 adjusted_light_position = light_position_or_extdir + light_normal*light_epsilon;
                    const vec3 ray_vec = adjusted_light_position - adjusted_hit_position;

                    ray_origin = adjusted_hit_position;
                    ray_distance = length(ray_vec);
                    ray_dir = ray_vec/ray_distance;
                }

                const bool is_occluded = trace_occlusion_ray(
                    ray_origin,
                    ray_dir,
                    ray_distance);
                if (!is_occluded) {
                    result_sum += result;
                }
            }
        }

        // sample BSDF
        {
            vec2 bsdf_rand_u01 = rand_u01(2*segment_index);

            // RR
            if (segment_index > 4) {
                const vec3 reflectance = get_reflectance(hit.bsdf_params);
                const float survive_prob = clamp(max_element(reflectance), .1f, .95f);
                if (!split_random_variable(survive_prob, bsdf_rand_u01.y)) {
                    break;
                }
                prev_sample /= survive_prob;
            }

            vec3 in_dir_ls;
            vec3 estimator;
            float solid_angle_pdf_or_negative;
            float sampled_roughness;
            sample_bsdf(
                get_bsdf_type(hit.info),
                out_dir_ls,
                hit.bsdf_params,
                bsdf_rand_u01,
                in_dir_ls,
                estimator,
                solid_angle_pdf_or_negative,
                roughness_acc);
            
            const mat3 hit_basis = basis_from_z_axis(get_dir(hit.shading_normal));
            const vec3 in_dir = normalize(hit_basis * in_dir_ls);
            roughness_acc = max(roughness_acc, sampled_roughness);

            prev_position = hit.position_or_extdir;
            prev_geom_normal_packed = hit.geom_normal;
            prev_epsilon = copysign(get_epsilon(hit.info, LOG2_EPSILON_FACTOR), in_dir_ls.z);
            prev_in_dir = in_dir;
            prev_in_solid_angle_pdf_or_negative = solid_angle_pdf_or_negative;
            prev_sample *= estimator;
        }
    }

    if (any(isinf(result_sum)) || any(isnan(result_sum))) {
        // skip
    } else {
        imageStore(g_result[0], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.x, 0, 0, 0));
        imageStore(g_result[1], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.y, 0, 0, 0));
        imageStore(g_result[2], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.z, 0, 0, 0));
    }
}
