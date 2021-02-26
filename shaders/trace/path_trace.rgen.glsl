#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "occlusion_common.glsl"
#include "sampler.glsl"
#include "tone_map.glsl"
#include "light_common.glsl"
#include "ggx.glsl"
#include "fresnel.glsl"
#include "spectrum.glsl"

#include "disc_light.glsl"
#include "dome_light.glsl"
#include "quad_light.glsl"
#include "solid_angle_light.glsl"
#include "sphere_light.glsl"

#include "diffuse_bsdf.glsl"
#include "mirror_bsdf.glsl"
#include "smooth_dielectric_bsdf.glsl"
#include "smooth_plastic_bsdf.glsl"
#include "rough_conductor_bsdf.glsl"
#include "rough_dielectric_bsdf.glsl"
#include "rough_plastic_bsdf.glsl"

#define PATH_TRACE_FLAG_ACCUMULATE_ROUGHNESS                0x01
#define PATH_TRACE_FLAG_ALLOW_LIGHT_SAMPLING                0x02
#define PATH_TRACE_FLAG_ALLOW_BSDF_SAMPLING                 0x04
#define PATH_TRACE_FLAG_SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE    0x08
#define PATH_TRACE_FLAG_PLANAR_LIGHTS_ARE_TWO_SIDED         0x10

#define MIS_HEURISTIC_NONE      0
#define MIS_HEURISTIC_BALANCE   1
#define MIS_HEURISTIC_POWER2    2

layout(set = 0, binding = 0, scalar) uniform PathTraceUniforms {
    LightInfoTable light_info_table;
    LightAliasTable light_alias_table;
    uint64_t light_params_base;
    uint sampled_light_count;
    uint external_light_begin;
    uint external_light_end;
    mat4x3 world_from_camera;
    vec2 fov_size_at_unit_z;
    uint sample_index;
    uint max_segment_count;
    uint mis_heuristic;
    uint sequence_type;
    uint flags;
} g_path_trace;

layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, rg32f) uniform restrict readonly image2D g_pmj_samples;
layout(set = 0, binding = 3, rgba32ui) uniform restrict readonly uimage2D g_sobol_samples;
layout(set = 0, binding = 4) uniform sampler1DArray g_illuminants;
layout(set = 0, binding = 5) uniform sampler2D g_smits_table;
layout(set = 0, binding = 6, r32f) uniform restrict writeonly image2D g_result;

#define LOG2_EPSILON_FACTOR     (-18)

#include "sequence.glsl"

vec4 rand_u01(uint seq_index)
{
    return rand_u01(gl_LaunchIDEXT.xy, seq_index, g_path_trace.sample_index, g_path_trace.sequence_type);
}

uint sample_uniform_discrete(uint count, inout float u01)
{
    const float index_flt = float(count)*u01;
    const uint index = min(uint(index_flt), count - 1);
    u01 = index_flt - float(index);
    return index;
}

EXTEND_PAYLOAD(g_extend);
OCCLUSION_PAYLOAD(g_occlusion);

float illuminant(float wavelength, vec3 emission)
{
    const float power = smits_power_from_rec709(wavelength, emission, g_smits_table);
    const float wavelength_u = unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelength);
    const float illum_index = 1.f;
    return power*texture(g_illuminants, vec2(wavelength_u, illum_index)).x;
}

void evaluate_bsdf(
    float wavelength,
    uint bsdf_type,
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out float f,
    out float solid_angle_pdf)
{
    vec3 f_tmp;
#define CALL(F) F(out_dir, in_dir, params, f_tmp, solid_angle_pdf)
    switch (bsdf_type) {
        case BSDF_TYPE_DIFFUSE:             CALL(diffuse_bsdf_eval); break;
        case BSDF_TYPE_MIRROR:              break;
        case BSDF_TYPE_SMOOTH_DIELECTRIC:   break;
        case BSDF_TYPE_ROUGH_DIELECTRIC:    CALL(rough_dielectric_bsdf_eval); break;
        case BSDF_TYPE_SMOOTH_PLASTIC:      CALL(smooth_plastic_bsdf_eval); break;
        case BSDF_TYPE_ROUGH_PLASTIC:       CALL(rough_plastic_bsdf_eval); break;
        case BSDF_TYPE_ROUGH_CONDUCTOR:     CALL(rough_conductor_bsdf_eval); break;
    }
#undef CALL
    f = f_tmp.x;
}

void sample_bsdf(
    float wavelength,
    uint bsdf_type,
    vec3 out_dir,
    BsdfParams params,
    vec3 bsdf_rand_u01,
    out vec3 in_dir,
    out float estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    vec3 estimator_tmp;
#define CALL(F) F(out_dir, params, bsdf_rand_u01, in_dir, estimator_tmp, solid_angle_pdf_or_negative, path_max_roughness)
    switch (bsdf_type) {
        case BSDF_TYPE_DIFFUSE:             CALL(diffuse_bsdf_sample); break;
        case BSDF_TYPE_MIRROR:              CALL(mirror_bsdf_sample); break;
        case BSDF_TYPE_SMOOTH_DIELECTRIC:   CALL(smooth_dielectric_bsdf_sample); break;
        case BSDF_TYPE_ROUGH_DIELECTRIC:    CALL(rough_dielectric_bsdf_sample); break;
        case BSDF_TYPE_SMOOTH_PLASTIC:      CALL(smooth_plastic_bsdf_sample); break;
        case BSDF_TYPE_ROUGH_PLASTIC:       CALL(rough_plastic_bsdf_sample); break;
        case BSDF_TYPE_ROUGH_CONDUCTOR:     CALL(rough_conductor_bsdf_sample); break;
    }
#undef CALL
    estimator = estimator_tmp.x;
}

void sample_single_light(
    float wavelength,
    uint light_index,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_position_or_extdir,
    out Normal32 light_normal,
    out float light_emission,
    out float light_solid_angle_pdf,
    out bool light_is_external,
    out float light_epsilon)
{
    LightInfoEntry light_info = g_path_trace.light_info_table.entries[light_index];
    const float selection_pdf = light_info.probability;

    const uint64_t params_addr = g_path_trace.light_params_base + light_info.params_offset;

    vec3 emission;
    float solid_angle_pdf_and_ext_bit;
    float unit_scale;
    switch (light_info.light_type) {
#define PARAMS  target_position, target_normal, light_rand_u01, light_position_or_extdir, light_normal, emission, solid_angle_pdf_and_ext_bit, unit_scale
        case LIGHT_TYPE_QUAD: {
            PlanarLightParamsBuffer buf = PlanarLightParamsBuffer(params_addr);
            const bool is_two_sided = (g_path_trace.flags & PATH_TRACE_FLAG_PLANAR_LIGHTS_ARE_TWO_SIDED) != 0;
            quad_light_sample(buf.params, is_two_sided, PARAMS);
        } break;
        case LIGHT_TYPE_DISC: {
            PlanarLightParamsBuffer buf = PlanarLightParamsBuffer(params_addr);
            const bool is_two_sided = (g_path_trace.flags & PATH_TRACE_FLAG_PLANAR_LIGHTS_ARE_TWO_SIDED) != 0;
            disc_light_sample(buf.params, is_two_sided, PARAMS);
        } break;
        case LIGHT_TYPE_SPHERE: {
            SphereLightParamsBuffer buf = SphereLightParamsBuffer(params_addr);
            const bool sample_solid_angle = (g_path_trace.flags & PATH_TRACE_FLAG_SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE) != 0;
            sphere_light_sample(buf.params, sample_solid_angle, PARAMS);
        } break;
        case LIGHT_TYPE_SOLID_ANGLE: {
            SolidAngleLightParamsBuffer buf = SolidAngleLightParamsBuffer(params_addr);
            solid_angle_light_sample(buf.params, PARAMS);
        } break;
#undef PARAMS
    }

    light_emission = illuminant(wavelength, emission);
    light_solid_angle_pdf = abs(solid_angle_pdf_and_ext_bit) * selection_pdf;
    light_is_external = sign_bit_set(solid_angle_pdf_and_ext_bit);
    light_epsilon = ldexp(unit_scale, LOG2_EPSILON_FACTOR);    
}

void sample_all_lights(
    float wavelength,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_position_or_extdir,
    out Normal32 light_normal,
    out float light_emission,
    out float light_solid_angle_pdf,
    out bool light_is_external,
    out float light_epsilon)
{
    // pick a light source
    const uint entry_index = sample_uniform_discrete(g_path_trace.sampled_light_count, light_rand_u01.x);
    const LightAliasEntry entry = g_path_trace.light_alias_table.entries[entry_index];
    uint light_index;
    if (split_random_variable(entry.split, light_rand_u01.y)) {
        light_index = entry.indices & 0xffffU;
    } else {
        light_index = entry.indices >> 16;
    }

    // sample this light
    sample_single_light(
        wavelength,
        light_index,
        target_position,
        target_normal,
        light_rand_u01,
        light_position_or_extdir,
        light_normal,
        light_emission,
        light_solid_angle_pdf,
        light_is_external,
        light_epsilon);
}

void evaluate_single_light(
    float wavelength,
    uint light_index,
    vec3 target_position,
    vec3 light_position_or_extdir,
    out float light_emission,
    out float light_solid_angle_pdf)
{
    LightInfoEntry light_info = g_path_trace.light_info_table.entries[light_index];
    const float selection_pdf = light_info.probability;

    const uint64_t params_addr = g_path_trace.light_params_base + light_info.params_offset;

    vec3 emission;
    float solid_angle_pdf;
    switch (light_info.light_type) {
#define PARAMS  target_position, light_position_or_extdir, emission, solid_angle_pdf
        case LIGHT_TYPE_QUAD:
        case LIGHT_TYPE_DISC: {
            PlanarLightParamsBuffer buf = PlanarLightParamsBuffer(params_addr);
            const bool is_two_sided = (g_path_trace.flags & PATH_TRACE_FLAG_PLANAR_LIGHTS_ARE_TWO_SIDED) != 0;
            planar_light_eval(buf.params, is_two_sided, PARAMS);
        } break;
        case LIGHT_TYPE_SPHERE: {
            SphereLightParamsBuffer buf = SphereLightParamsBuffer(params_addr);
            const bool sample_solid_angle = (g_path_trace.flags & PATH_TRACE_FLAG_SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE) != 0;
            sphere_light_eval(buf.params, sample_solid_angle, PARAMS);
        } break;
        case LIGHT_TYPE_DOME: {
            DomeLightParamsBuffer buf = DomeLightParamsBuffer(params_addr);
            dome_light_eval(buf.params, PARAMS);
        } break;
        case LIGHT_TYPE_SOLID_ANGLE: {
            SolidAngleLightParamsBuffer buf = SolidAngleLightParamsBuffer(params_addr);
            solid_angle_light_eval(buf.params, PARAMS);
        } break;
#undef PARAMS
    }

    light_emission = illuminant(wavelength, emission);
    light_solid_angle_pdf = solid_angle_pdf * selection_pdf;
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

    float wavelength;
    vec3 prev_position;
    Normal32 prev_geom_normal_packed;
    float prev_epsilon;
    vec3 prev_in_dir;
    float prev_in_solid_angle_pdf_or_negative;
    float prev_sample;
    float path_max_roughness;

    // sample the camera
    {
        const vec4 pixel_rand_u01 = rand_u01(0);
        const vec2 fov_uv = (vec2(gl_LaunchIDEXT.xy) + pixel_rand_u01.xy)/vec2(gl_LaunchSizeEXT);
        const vec3 ray_dir_ls = normalize(vec3(g_path_trace.fov_size_at_unit_z*(.5f - fov_uv), 1.f));
        const vec3 ray_dir = g_path_trace.world_from_camera * vec4(ray_dir_ls, 0.f);
        
        const float fov_area_at_unit_z = mul_elements(g_path_trace.fov_size_at_unit_z);
        const float cos_theta = ray_dir_ls.z;
        const float cos_theta2 = cos_theta*cos_theta;
        const float cos_theta3 = cos_theta2*cos_theta;
        const float solid_angle_pdf = 1.f/(fov_area_at_unit_z*cos_theta3);

        const float importance = 1.f/fov_area_at_unit_z;
        const float sensor_area_pdf = 1.f/fov_area_at_unit_z;

        wavelength = mix(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, pixel_rand_u01.z);
        const float wavelength_pdf = 1.f/(SMITS_WAVELENGTH_MAX - SMITS_WAVELENGTH_MIN);

        prev_position = g_path_trace.world_from_camera[3];
        prev_geom_normal_packed = make_normal32(g_path_trace.world_from_camera[2]);
        prev_epsilon = 0.f;
        prev_in_dir = ray_dir;
        prev_in_solid_angle_pdf_or_negative = solid_angle_pdf;
        prev_sample = importance/(sensor_area_pdf*wavelength_pdf);
        path_max_roughness = 0.f;
    }

    // trace a path from the camera
    float result_sum = 0.f;
    uint segment_index = 0;
    for (;;) {
        // extend the path using the sampled (incoming) direction
        ExtendPayload hit = trace_extend_ray(
            prev_position + prev_epsilon*get_dir(prev_geom_normal_packed),
            prev_in_dir);
        ++segment_index;

        // compute reflectance for this wavelength
        const float reflectance_w = smits_power_from_rec709(
            wavelength,
            get_reflectance(hit.bsdf_params),
            g_smits_table);
        hit.bsdf_params = replace_reflectance(hit.bsdf_params, vec3(reflectance_w, 0.f, 0.f));

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
        if (allow_bsdf_sampling || path_max_roughness == 0.f) {
            if (has_light(hit.info)) {
                light_index_begin = get_light_index(hit.info);
                light_index_end = light_index_begin + 1;
            } else if (!has_surface(hit.info)) {
                light_index_begin = g_path_trace.external_light_begin;
                light_index_end = g_path_trace.external_light_end;
            }
        }
        for (uint light_index = light_index_begin; light_index != light_index_end; ++light_index) {
            // evaluate the light here
            const vec3 light_position_or_extdir = hit.position_or_extdir;
            float light_emission;
            float light_solid_angle_pdf;
            evaluate_single_light(
                wavelength,
                light_index,
                prev_position,
                light_position_or_extdir,
                light_emission,
                light_solid_angle_pdf);

            // compute MIS weight
            const bool light_can_be_sampled = allow_light_sampling && (light_index < g_path_trace.sampled_light_count);
            const bool prev_is_delta = sign_bit_set(prev_in_solid_angle_pdf_or_negative);
            const bool can_be_sampled = (segment_index > 1) && light_can_be_sampled && !prev_is_delta;
            const float other_ratio = can_be_sampled ? mis_ratio(light_solid_angle_pdf / prev_in_solid_angle_pdf_or_negative) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // accumulate this sample
            result_sum += prev_sample * light_emission * mis_weight;
        }

        // end the ray if we didn't hit any surface
        if (!has_surface(hit.info) || get_bsdf_type(hit.info) == BSDF_TYPE_NONE) {
            break;
        }
        if (segment_index >= g_path_trace.max_segment_count) {
            break;
        }

        // rewrite the BRDF to ensure roughness never reduces when extending an eye path
        if (accumulate_roughness) {
            path_max_roughness = max(path_max_roughness, get_roughness(hit.bsdf_params));
            switch (get_bsdf_type(hit.info)) {
                case BSDF_TYPE_MIRROR:
                case BSDF_TYPE_ROUGH_CONDUCTOR:
                    if (path_max_roughness != 0.f) {
                        const uint bsdf_type = (path_max_roughness < 1.f) ? BSDF_TYPE_ROUGH_CONDUCTOR : BSDF_TYPE_DIFFUSE;
                        hit.info = replace_bsdf_type(hit.info, bsdf_type);
                        hit.bsdf_params = replace_roughness(hit.bsdf_params, path_max_roughness);
                    }
                    break;
                case BSDF_TYPE_SMOOTH_DIELECTRIC:
                    if (path_max_roughness != 0.f) {
                        hit.info = replace_bsdf_type(hit.info, BSDF_TYPE_ROUGH_DIELECTRIC);
                        hit.bsdf_params = replace_roughness(hit.bsdf_params, path_max_roughness);
                    }
                    break;
                case BSDF_TYPE_DIFFUSE:
                    // nothing to do
                    break;
                case BSDF_TYPE_ROUGH_PLASTIC:
                case BSDF_TYPE_SMOOTH_PLASTIC:
                    if (path_max_roughness != 0.f) {
                        const uint bsdf_type = (path_max_roughness < 1.f) ? BSDF_TYPE_ROUGH_PLASTIC : BSDF_TYPE_DIFFUSE;
                        hit.info = replace_bsdf_type(hit.info, bsdf_type);
                        hit.bsdf_params = replace_roughness(hit.bsdf_params, path_max_roughness);
                    }
                    break;
            }
        }
        if (get_bsdf_type(hit.info) == BSDF_TYPE_NONE) {
            break;
        }

        // unpack the BRDF
        const vec3 out_dir_ls = normalize(-prev_in_dir * basis_from_z_axis(get_dir(hit.shading_normal)));

        // sample a light source
        const bool hit_is_always_delta = bsdf_is_always_delta(get_bsdf_type(hit.info));
        if (g_path_trace.sampled_light_count != 0 && allow_light_sampling && !hit_is_always_delta) {
            // sequence for light sampling
            const vec2 light_rand_u01 = rand_u01(2*segment_index - 1).xy;

            // sample from all light sources
            vec3 light_position_or_extdir;
            Normal32 light_normal;
            float light_emission;
            float light_solid_angle_pdf;
            bool light_is_external;
            float light_epsilon;
            sample_all_lights(
                wavelength,
                hit.position_or_extdir,
                get_dir(hit.shading_normal),
                light_rand_u01,
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
            float hit_f;
            float hit_solid_angle_pdf;
            if (in_cos_theta > 0.f || bsdf_has_transmission(bsdf_type)) {
                evaluate_bsdf(
                    wavelength,
                    bsdf_type,
                    out_dir_ls,
                    in_dir_ls,
                    hit.bsdf_params,
                    hit_f,
                    hit_solid_angle_pdf);
            } else {
                hit_f = 0.f;
                hit_solid_angle_pdf = 0.f;
            }
            
            // compute MIS weight
            const bool light_can_be_hit = allow_bsdf_sampling;
            const float other_ratio = light_can_be_hit ? mis_ratio(hit_solid_angle_pdf/light_solid_angle_pdf) : 0.f;
            const float mis_weight = 1.f/(1.f + other_ratio);

            // compute the sample assuming the ray is not occluded
            const float result = prev_sample * hit_f * (mis_weight*abs(in_cos_theta)/light_solid_angle_pdf) * light_emission;

            // trace an occlusion ray if necessary
            if (result > 0.f) {
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
                    const vec3 adjusted_light_position = light_position_or_extdir + get_dir(light_normal)*light_epsilon;
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
            vec3 bsdf_rand_u01 = rand_u01(2*segment_index).xyz;

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
            float estimator;
            float solid_angle_pdf_or_negative;
            sample_bsdf(
                wavelength,
                get_bsdf_type(hit.info),
                out_dir_ls,
                hit.bsdf_params,
                bsdf_rand_u01,
                in_dir_ls,
                estimator,
                solid_angle_pdf_or_negative,
                path_max_roughness);
            
            const mat3 hit_basis = basis_from_z_axis(get_dir(hit.shading_normal));
            const vec3 in_dir = normalize(hit_basis * in_dir_ls);

            prev_position = hit.position_or_extdir;
            prev_geom_normal_packed = hit.geom_normal;
            prev_epsilon = copysign(get_epsilon(hit.info, LOG2_EPSILON_FACTOR), in_dir_ls.z);
            prev_in_dir = in_dir;
            prev_in_solid_angle_pdf_or_negative = solid_angle_pdf_or_negative;
            prev_sample *= estimator;
        }
    }

    imageStore(g_result, ivec2(gl_LaunchIDEXT.xy), vec4(result_sum, 0, 0, 0));
}
