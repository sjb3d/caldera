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
#include "rand_common.glsl"

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

#define PLASTIC_F0              0.04f

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

bool split_random_variable(float accept_probability, inout float u01)
{
    const bool is_accept = (u01 <= accept_probability);
    if (is_accept) {
        u01 /= accept_probability;
    } else {
        u01 -= accept_probability;
        u01 /= (1.f - accept_probability);
    }
    return is_accept;
}

// approximation from http://c0de517e.blogspot.com/2019/08/misunderstanding-multilayering-diffuse.html
float remaining_diffuse_strength(float n_dot_v, float roughness)
{
    return mix(1.f - schlick_fresnel(PLASTIC_F0, n_dot_v), 1.f - PLASTIC_F0, roughness);
}

// reference: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float fresnel_dieletric_reflection(float cos_theta, float eta)
{
    const float c = cos_theta;
    const float temp = eta*eta + c*c - 1.f;
    if (temp < 0.f) {
        return 1.f;
    }

    const float g = sqrt(temp);
    return .5f
        *square((g - c)/(g + c))
        *(1.f + square(((g + c)*c - 1.f)/((g - c)*c + 1.f)));
}

vec3 refract(vec3 v, float eta)
{
    // Snell's law: sin_theta_i = sin_theta_t * eta
    const float cos_theta_i = abs(v.z);
    const float sin2_theta_i = 1.f - cos_theta_i*cos_theta_i;
    const float sin2_theta_t = sin2_theta_i/square(eta);
    const float cos_theta_t = sqrt(max(1.f - sin2_theta_t, 0.f));
    return normalize(vec3(0.f, 0.f, cos_theta_i/eta - cos_theta_t) - v/eta);
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
    const bool accumulate_roughness = ((g_path_trace.flags & PATH_TRACE_FLAG_ACCUMULATE_ROUGHNESS) != 0);

    vec3 prev_position;
    ExtendPackedNormal prev_geom_normal_packed;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_in_solid_angle_pdf; // negative for delta
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
        prev_geom_normal_packed = make_packed_normal(g_path_trace.world_from_camera[2]);
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_in_solid_angle_pdf = solid_angle_pdf;
        prev_sample = importance/sensor_area_pdf;
        roughness_acc = 0.f;
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
            prev_position + prev_epsilon*get_dir(prev_geom_normal_packed),
            0.f,
            prev_in_dir,
            FLT_INF,
            EXTEND_PAYLOAD_INDEX);
        ++segment_index;

        // reject implausible shading normals
        {
            const vec3 hit_shading_normal_vec = get_vec(g_extend.shading_normal);
            if (has_surface(g_extend.hit) && dot(hit_shading_normal_vec, prev_in_dir) >= 0.f) {
                break;
            }
        }

        // handle ray hitting a light source
        uint light_index_begin = 0;
        uint light_index_end = 0;
        if (allow_bsdf_sampling || roughness_acc == 0.f) {
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
            const bool prev_is_delta = sign_bit_set(prev_in_solid_angle_pdf);
            const bool can_be_sampled = (segment_index > 1) && light_can_be_sampled && !prev_is_delta;
            const float other_ratio = can_be_sampled ? mis_ratio(light_solid_angle_pdf / prev_in_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // accumulate this sample
            result_sum += prev_sample * light_emission * mis_weight;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(g_extend.hit)) {
            break;
        }
        if (segment_index >= g_path_trace.max_segment_count) {
            break;
        }

        // rewrite the BRDF to ensure roughness never reduces when extending an eye path
        if (accumulate_roughness) {
            const float orig_roughness = get_roughness(g_extend.hit);
            const float p = 4.f;
            roughness_acc = min(pow(pow(roughness_acc, p) + pow(orig_roughness, p), 1.f/p), 1.f);

            switch (get_bsdf_type(g_extend.hit)) {
                case BSDF_TYPE_MIRROR:
                case BSDF_TYPE_CONDUCTOR:
                    if (roughness_acc != 0.f) {
                        g_extend.hit = replace_hit_data(g_extend.hit, BSDF_TYPE_CONDUCTOR, roughness_acc);
                    }
                    break;
                case BSDF_TYPE_DIELECTRIC:
                    // TODO: degrade to rough dielectric
                    break;
                case BSDF_TYPE_DIFFUSE:
                    // nothing to do
                    break;
                case BSDF_TYPE_PLASTIC:
                case BSDF_TYPE_SMOOTH_PLASTIC:
                    if (roughness_acc != 0.f) {
                        g_extend.hit = replace_hit_data(g_extend.hit, BSDF_TYPE_PLASTIC, roughness_acc);
                    }
                    break;
            }
        }

        // unpack the BRDF
        const vec3 hit_reflectance = sample_from_rec709(get_reflectance(g_extend.hit));
        const float hit_roughness = get_roughness(g_extend.hit);

        const vec3 hit_position = g_extend.position_or_extdir;
        const mat3 hit_basis = basis_from_z_axis(get_dir(g_extend.shading_normal));
        const vec3 hit_shading_normal = hit_basis[2];
        const vec3 hit_geom_normal = get_dir(g_extend.geom_normal);
        const bool hit_is_always_delta = bsdf_is_always_delta(get_bsdf_type(g_extend.hit));

        const vec3 out_dir_ls = normalize(-prev_in_dir * hit_basis);

        const float hit_epsilon = get_epsilon(g_extend.hit, LOG2_EPSILON_FACTOR);

        // sample a light source
        if (g_light.sampled_count != 0 && allow_light_sampling && !hit_is_always_delta) {
            // sample from all light sources
            vec3 light_position_or_extdir;
            vec3 light_normal;
            vec3 light_emission;
            float light_solid_angle_pdf;
            bool light_is_external;
            float light_epsilon;
            sample_all_lights(
                hit_position,
                hit_shading_normal,
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
            if (in_cos_theta > 0.f) {
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
                        const float h_dot_v = abs(dot(h, v));
                        const vec3 r0 = hit_reflectance;
                        hit_f = ggx_brdf(r0, h, h_dot_v, v, l, alpha);

                        hit_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
                    } break;

                    case BSDF_TYPE_PLASTIC: {
                        const float diffuse_strength = remaining_diffuse_strength(out_dir_ls.z, hit_roughness);
                        const vec2 alpha = vec2(hit_roughness*hit_roughness);

                        const vec3 v = out_dir_ls;
                        const vec3 l = in_dir_ls;
                        const vec3 h = normalize(v + l);
                        const float h_dot_v = abs(dot(h, v));
                        const float spec_f = ggx_brdf(PLASTIC_F0, h, h_dot_v, v, l, alpha);

                        const vec3 diff_f = hit_reflectance*(diffuse_strength/PI);

                        const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_cos_theta);
                        const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
                        const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_strength);

                        hit_f = diff_f + vec3(spec_f);
                        hit_solid_angle_pdf = combined_solid_angle_pdf;
                    } break;

                    case BSDF_TYPE_SMOOTH_PLASTIC: {
                        const float diffuse_strength = remaining_diffuse_strength(out_dir_ls.z, 0.f);

                        const vec3 diff_f = hit_reflectance*(diffuse_strength/PI);
                        const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_cos_theta);

                        hit_f = diff_f;
                        hit_solid_angle_pdf = diffuse_strength*diff_solid_angle_pdf;
                    } break;
                }
            }
            
            // compute MIS weight
            const bool light_can_be_hit = allow_bsdf_sampling;
            const float other_ratio = light_can_be_hit ? mis_ratio(hit_solid_angle_pdf/light_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // compute the sample assuming the ray is not occluded
            const vec3 result = prev_sample * hit_f * (mis_weight*abs(in_cos_theta)/light_solid_angle_pdf) * light_emission;

            // trace an occlusion ray if necessary
            if (any(greaterThan(result, vec3(0.f)))) {
                const vec3 adjusted_hit_position = hit_position + hit_geom_normal*hit_epsilon;

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
                case BSDF_TYPE_MIRROR: {
                    in_dir_ls = vec3(-out_dir_ls.xy, out_dir_ls.z);
                    estimator = hit_reflectance;
                    in_solid_angle_pdf = -1.f;
                } break;

                case BSDF_TYPE_DIELECTRIC: {
                    const float eta_front = 1.333f;
                    const float eta = is_front_hit(g_extend.hit) ? eta_front : (1.f/eta_front);
                    const float reflect_chance = fresnel_dieletric_reflection(out_dir_ls.z, eta);

                    if (bsdf_rand_u01.x > reflect_chance) {
                        in_dir_ls = refract(out_dir_ls, eta);
                    } else {
                        in_dir_ls = vec3(-out_dir_ls.xy, out_dir_ls.z);
                    }

                    estimator = hit_reflectance;
                    in_solid_angle_pdf = -1.f;
                } break;

                default:
                case BSDF_TYPE_DIFFUSE: {
                    in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);

                    const vec3 f = hit_reflectance/PI;
                    estimator = f/get_hemisphere_cosine_weighted_proj_pdf();

                    in_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_dir_ls.z);
                } break;

                case BSDF_TYPE_CONDUCTOR: {
                    const vec2 alpha = vec2(hit_roughness*hit_roughness);

                    const vec3 h = sample_vndf(out_dir_ls, alpha, bsdf_rand_u01);
                    in_dir_ls = reflect(-out_dir_ls, h);
                    in_dir_ls.z = abs(in_dir_ls.z);

                    const vec3 v = out_dir_ls;
                    const vec3 l = in_dir_ls;
                    const float h_dot_v = abs(dot(h, v));
                    const vec3 r0 = hit_reflectance;
                    estimator = ggx_vndf_sampled_estimator(r0, h_dot_v, v, l, alpha);

                    in_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
                } break;

                case BSDF_TYPE_PLASTIC: {
                    const float diffuse_strength = remaining_diffuse_strength(out_dir_ls.z, hit_roughness);
                    const bool sample_diffuse = split_random_variable(diffuse_strength, bsdf_rand_u01.x);

                    const vec2 alpha = vec2(hit_roughness*hit_roughness);

                    if (sample_diffuse) {
                        in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);
                    } else {
                        const vec3 h = sample_vndf(out_dir_ls, alpha, bsdf_rand_u01);
                        in_dir_ls = reflect(-out_dir_ls, h);
                        in_dir_ls.z = abs(in_dir_ls.z);
                    }

                    const vec3 v = out_dir_ls;
                    const vec3 l = in_dir_ls;
                    const vec3 h = normalize(v + l);
                    const float h_dot_v = abs(dot(h, v));
                    const float n_dot_l = l.z;

                    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
                    const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
                    const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_strength);

                    const float spec_f = ggx_brdf(PLASTIC_F0, h, h_dot_v, v, l, alpha);
                    const vec3 diff_f = hit_reflectance*(diffuse_strength/PI);
                    const vec3 f = vec3(spec_f) + diff_f;

                    estimator = f * n_dot_l / combined_solid_angle_pdf;
                    in_solid_angle_pdf = combined_solid_angle_pdf;
                } break;

                case BSDF_TYPE_SMOOTH_PLASTIC: {
                    const float diffuse_strength = remaining_diffuse_strength(out_dir_ls.z, 0.f);
                    const float spec_strength = 1.f - diffuse_strength;
    
                    const bool sample_diffuse = split_random_variable(diffuse_strength, bsdf_rand_u01.x);
                    if (sample_diffuse) {
                        in_dir_ls = sample_hemisphere_cosine_weighted(bsdf_rand_u01);
                        roughness_acc = 1.f;
                    } else {
                        in_dir_ls = vec3(-out_dir_ls.xy, out_dir_ls.z);
                    }

                    const float spec_proj_solid_angle_pdf = 1.f;

                    if (sample_diffuse) {
                        const float n_dot_l = in_dir_ls.z;
                        const vec3 diff_f = hit_reflectance*(diffuse_strength/PI);

                        estimator = diff_f/(diffuse_strength*get_hemisphere_cosine_weighted_proj_pdf());
                        in_solid_angle_pdf = diffuse_strength*get_hemisphere_cosine_weighted_pdf(n_dot_l);
                    } else {
                        const float n_dot_v = out_dir_ls.z;
                        const float spec_f = schlick_fresnel(PLASTIC_F0, n_dot_v);

                        estimator = vec3(spec_f)/spec_strength;
                        in_solid_angle_pdf = -1.f;
                    }
                } break;
            }
            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = hit_position;
            prev_geom_normal_packed = g_extend.geom_normal;
            prev_epsilon = copysign(hit_epsilon, in_dir_ls.z);
            prev_in_dir = in_dir;
            prev_in_solid_angle_pdf = in_solid_angle_pdf;
            prev_sample *= estimator;
        }
    }

    imageStore(g_result[0], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.x, 0, 0, 0));
    imageStore(g_result[1], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.y, 0, 0, 0));
    imageStore(g_result[2], ivec2(gl_LaunchIDEXT.xy), vec4(result_sum.z, 0, 0, 0));
}
