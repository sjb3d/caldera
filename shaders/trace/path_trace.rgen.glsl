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
#include "ggx.glsl"

#define PATH_TRACE_FLAG_USE_MAX_ROUGHNESS       0x01
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

#define PLASTIC_F0              0.05f

#define RENDER_COLOR_SPACE_REC709   0
#define RENDER_COLOR_SPACE_ACESCG   1

layout(set = 0, binding = 2) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 3, r16ui) uniform restrict readonly uimage2D g_samples;
layout(set = 0, binding = 4, r32f) uniform restrict image2D g_result_r;
layout(set = 0, binding = 5, r32f) uniform restrict image2D g_result_g;
layout(set = 0, binding = 6, r32f) uniform restrict image2D g_result_b;

#define LOG2_SEQUENCE_COUNT     10

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
    out vec3 light_position_or_extdir,
    out vec3 light_normal,
    out vec3 light_emission,
    out float light_solid_angle_pdf,
    out bool light_is_external,
    out float light_epsilon)
{
    g_light_sample.position_or_extdir = target_position;
    g_light_sample.normal = target_normal;
    g_light_sample.emission = vec3(rand_u01, 0.f);
    executeCallableEXT(LIGHT_SAMPLE_SHADER_INDEX(light_index), LIGHT_SAMPLE_CALLABLE_INDEX);
    light_position_or_extdir = g_light_sample.position_or_extdir;
    light_normal = g_light_sample.normal;
    light_emission = sample_from_rec709(g_light_sample.emission);
    light_solid_angle_pdf = abs(g_light_sample.solid_angle_pdf_and_extbit);
    light_is_external = sign_bit_set(g_light_sample.solid_angle_pdf_and_extbit);
    light_epsilon = ldexp(g_light_sample.unit_scale, LOG2_EPSILON_FACTOR);    
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
    const float sampled_count_flt = float(g_light.sampled_count);
    const float rand_light_flt = rand_u01.x*sampled_count_flt;
    const uint light_index = min(uint(rand_light_flt), g_light.sampled_count - 1);
    rand_u01.x = rand_light_flt - float(light_index);
    const float selection_pdf = 1.f/sampled_count_flt;

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
    g_light_eval.position_or_extdir = light_position_or_extdir;
    g_light_eval.emission = prev_position;
    executeCallableEXT(LIGHT_EVAL_SHADER_INDEX(light_index), LIGHT_EVAL_CALLABLE_INDEX);
    light_emission = sample_from_rec709(g_light_eval.emission);
    light_solid_angle_pdf = g_light_eval.solid_angle_pdf;
}

float get_light_selection_pdf()
{
    const float sampled_count_flt = float(g_light.sampled_count);
    return 1.f/sampled_count_flt;
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

void main()
{
    const bool allow_light_sampling = ((g_path_trace.flags & PATH_TRACE_FLAG_ALLOW_LIGHT_SAMPLING) != 0);
    const bool allow_bsdf_sampling = ((g_path_trace.flags & PATH_TRACE_FLAG_ALLOW_BSDF_SAMPLING) != 0);
    const bool use_max_roughness = ((g_path_trace.flags & PATH_TRACE_FLAG_USE_MAX_ROUGHNESS) != 0);

    vec3 prev_position;
    vec3 prev_normal;
    bool prev_is_delta;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_in_solid_angle_pdf;
    vec3 alpha;
    float max_roughness;

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
        prev_normal = g_path_trace.world_from_camera[2];
        prev_is_delta = false;
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_in_solid_angle_pdf = solid_angle_pdf;
        alpha = importance/sensor_area_pdf;
        max_roughness = 0.f;
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
        uint light_index_begin = 0;
        uint light_index_end = 0;
        if (allow_bsdf_sampling || segment_index == 1) {
            if (has_light(g_extend.hit)) {
                light_index_begin = get_light_index(g_extend.hit);
                light_index_end = light_index_begin + 1;
            } else if (!has_surface(g_extend.hit)) {
                light_index_begin = g_light.external_begin;
                light_index_end = g_light.external_end;
            }
        }
        for (uint light_index = light_index_begin; light_index != light_index_end; ++light_index) {
            // evaluate the light here
            const vec3 light_position_or_extdir = g_extend.position_or_extdir;
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
                light_solid_angle_pdf *= get_light_selection_pdf();
            }

            // compute MIS weight
            const bool can_be_sampled = (segment_index > 1) && light_can_be_sampled && !prev_is_delta;
            const float other_ratio = can_be_sampled ? mis_ratio(light_solid_angle_pdf / prev_in_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // accumulate this sample
            result_sum += alpha * light_emission * mis_weight;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(g_extend.hit)) {
            break;
        }
        if (segment_index >= g_path_trace.max_segment_count) {
            break;
        }

        // rewrite the BRDF to ensure roughness never reduces when extending an eye path
        if (use_max_roughness) {
            max_roughness = max(max_roughness, get_roughness(g_extend.hit));
            if (max_roughness == 1.f) {
                g_extend.hit = replace_hit_data(g_extend.hit, BSDF_TYPE_DIFFUSE, max_roughness);
            } else if (max_roughness != 0.f) {
                const uint orig_bsdf_type = get_bsdf_type(g_extend.hit);
                const uint new_bsdf_type = (orig_bsdf_type == BSDF_TYPE_MIRROR) ? BSDF_TYPE_DIFFUSE : orig_bsdf_type;
                g_extend.hit = replace_hit_data(g_extend.hit, new_bsdf_type, max_roughness);
            }
        }

        // unpack the BRDF
        const vec3 hit_reflectance = sample_from_rec709(get_reflectance(g_extend.hit));
        const float hit_roughness = get_roughness(g_extend.hit);

        const vec3 hit_position = g_extend.position_or_extdir;
        const mat3 hit_basis = basis_from_z_axis(normalize(vec_from_oct32(g_extend.normal_oct32)));
        const vec3 hit_normal = hit_basis[2];
        const bool hit_is_delta = (get_bsdf_type(g_extend.hit) == BSDF_TYPE_MIRROR);

        const vec3 out_dir_ls = normalize(-prev_in_dir * hit_basis);

        const float hit_epsilon = get_epsilon(g_extend.hit, LOG2_EPSILON_FACTOR);

        // sample a light source
        if (g_light.sampled_count != 0 && allow_light_sampling && !hit_is_delta) {
            // sample from all light sources
            vec3 light_position_or_extdir;
            vec3 light_normal;
            vec3 light_emission;
            float light_solid_angle_pdf;
            bool light_is_external;
            float light_epsilon;
            sample_all_lights(
                hit_position,
                hit_normal,
                rand_u01(2*segment_index - 1),
                light_position_or_extdir,
                light_normal,
                light_emission,
                light_solid_angle_pdf,
                light_is_external,
                light_epsilon);

            vec3 in_dir_ls;
            if (light_is_external) {
                in_dir_ls = normalize(light_position_or_extdir*hit_basis);
            } else {
                in_dir_ls = normalize((light_position_or_extdir - hit_position)*hit_basis);
            }

            // evaluate the BRDF
            const float in_cos_theta = in_dir_ls.z;
            vec3 hit_f = vec3(0.f);
            float hit_solid_angle_pdf = 0.f;
            if (sign_bits_match(in_dir_ls.z, out_dir_ls.z)) {
                switch (get_bsdf_type(g_extend.hit)) {
                    default:
                    case BSDF_TYPE_DIFFUSE: {
                        hit_f = hit_reflectance/PI;
                        hit_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_cos_theta);
                    } break;

                    case BSDF_TYPE_CONDUCTOR: {
                        const vec2 alpha = vec2(hit_roughness*hit_roughness);

                        const vec3 v = out_dir_ls;
                        const vec3 l = in_dir_ls;
                        const vec3 h = normalize(v + l);
                        const float h_dot_v = dot(h, v);
                        const vec3 r0 = hit_reflectance;
                        hit_f = ggx_brdf(r0, h, h_dot_v, v, l, alpha);

                        const float n_dot_v = v.z;
                        hit_solid_angle_pdf = ggx_vndf_pdf(v, h, h_dot_v, alpha) / (4.f * n_dot_v);
                    } break;

                    case BSDF_TYPE_PLASTIC: {
                        const vec3 diff_f = hit_reflectance*(1.f - PLASTIC_F0)/PI;
                        const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_cos_theta);

                        const vec2 alpha = vec2(hit_roughness*hit_roughness);

                        const vec3 v = out_dir_ls;
                        const vec3 l = in_dir_ls;
                        const vec3 h = normalize(v + l);
                        const float h_dot_v = dot(h, v);
                        const float spec_f = ggx_brdf(PLASTIC_F0, h, h_dot_v, v, l, alpha);

                        const float n_dot_v = v.z;
                        const float spec_solid_angle_pdf = ggx_vndf_pdf(v, h, h_dot_v, alpha) / (4.f * n_dot_v);

                        hit_f = diff_f + vec3(spec_f);
                        hit_solid_angle_pdf = .5f*(diff_solid_angle_pdf + spec_solid_angle_pdf);
                    } break;
                }
            }
            
            // compute MIS weight
            const bool light_can_be_hit = allow_bsdf_sampling;
            const float other_ratio = light_can_be_hit ? mis_ratio(hit_solid_angle_pdf/light_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // compute the sample assuming the ray is not occluded
            const vec3 result = alpha * hit_f * (mis_weight*abs(in_cos_theta)/light_solid_angle_pdf) * light_emission;

            // trace an occlusion ray if necessary
            if (any(greaterThan(result, vec3(0.f)))) {
                const vec3 adjusted_hit_position = hit_position + hit_normal*hit_epsilon;

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

                if (g_occlusion.is_occluded == 0) {
                    result_sum += result;
                }
            }
        }

        // TODO: RR once indirect

        // sample BSDF
        {
            vec2 bsdf_rand_u01 = rand_u01(2*segment_index);
            vec3 in_dir_ls;
            vec3 estimator;
            float in_solid_angle_pdf;
            switch (get_bsdf_type(g_extend.hit)) {
                default:
                case BSDF_TYPE_DIFFUSE: {
                    in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);

                    const vec3 f = hit_reflectance/PI;
                    estimator = f/get_hemisphere_cosine_weighted_proj_pdf();

                    in_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_dir_ls.z);
                } break;

                case BSDF_TYPE_MIRROR: {
                    in_dir_ls = vec3(-out_dir_ls.xy, out_dir_ls.z);
                    estimator = hit_reflectance;
                    in_solid_angle_pdf = 1.f/abs(in_dir_ls.z);
                } break;

                case BSDF_TYPE_CONDUCTOR: {
                    const vec2 alpha = vec2(hit_roughness*hit_roughness);

                    const vec3 h = sample_ggx_vndf(out_dir_ls, alpha, bsdf_rand_u01);
                    in_dir_ls = reflect(-out_dir_ls, h);
                    in_dir_ls.z = abs(in_dir_ls.z);

                    const vec3 v = out_dir_ls;
                    const vec3 l = in_dir_ls;
                    const float h_dot_v = dot(h, v);
                    const vec3 r0 = hit_reflectance;
                    estimator = ggx_vndf_sampled_estimator(r0, h_dot_v, v, l, alpha);

                    in_solid_angle_pdf = ggx_vndf_pdf(v, h, h_dot_v, alpha);
                } break;

                case BSDF_TYPE_PLASTIC: {
                    const float rand_layer_flt = 2.f*bsdf_rand_u01.x;
                    bsdf_rand_u01.x = rand_layer_flt - floor(rand_layer_flt);
                    if (rand_layer_flt > 1.f) {
                        in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);

                        const vec3 f = hit_reflectance*(1.f - PLASTIC_F0)/PI;
                        estimator = f/get_hemisphere_cosine_weighted_proj_pdf();

                        in_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_dir_ls.z);
                    } else {
                        const vec2 alpha = vec2(hit_roughness*hit_roughness);

                        const vec3 h = sample_ggx_vndf(out_dir_ls, alpha, bsdf_rand_u01);
                        in_dir_ls = reflect(-out_dir_ls, h);
                        in_dir_ls.z = abs(in_dir_ls.z);

                        const vec3 v = out_dir_ls;
                        const vec3 l = in_dir_ls;
                        const float h_dot_v = dot(h, v);
                        estimator = vec3(ggx_vndf_sampled_estimator(PLASTIC_F0, h_dot_v, v, l, alpha));

                        in_solid_angle_pdf = ggx_vndf_pdf(v, h, h_dot_v, alpha);
                    }
                    const float selection_pdf = .5f;
                    estimator /= selection_pdf;
                    in_solid_angle_pdf *= selection_pdf;
                } break;
            }
            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = hit_position;
            prev_normal = hit_normal;
            prev_is_delta = hit_is_delta;
            prev_epsilon = hit_epsilon;
            prev_in_dir = in_dir;
            prev_in_solid_angle_pdf = in_solid_angle_pdf;
            alpha *= estimator;
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
