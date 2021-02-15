#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

LIGHT_UNIFORM_DATA(g_light);

SPHERE_LIGHT_RECORD(g_record);

LIGHT_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 light_position = g_eval.position_or_extdir;
    const vec3 target_position = g_eval.emission;

    const vec3 light_normal = normalize(light_position - g_record.centre_ws);
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    const vec3 emission = (facing_term > 0.f) ? g_record.emission : vec3(0.f);

    g_eval.emission = emission;
    
    if (g_light.sample_sphere_solid_angle != 0) {
        const vec3 centre_from_target = g_record.centre_ws - target_position;
        const float sin_theta = g_record.radius_ws/length(centre_from_target);

        float solid_angle;
        if (sin_theta > SPHERE_LIGHT_SIN_THETA_MIN) {
            // accurate solid angle on a sphere
            const float cos_theta = sqrt(max(0.f, 1.f - sin_theta*sin_theta));
            solid_angle = 2.f*PI*(1.f - cos_theta);
        } else {
            // approximate with a disc
            solid_angle = PI*sin_theta*sin_theta;
        }

        g_eval.solid_angle_pdf = 1.f/solid_angle;
    } else {
        const float distance_sq = dot(target_from_light, target_from_light);
        const float area_ws = 4.f * PI * g_record.radius_ws * g_record.radius_ws;
        const float solid_angle_pdf = solid_angle_pdf_from_area_pdf(1.f/area_ws, facing_term, distance_sq);

        g_eval.solid_angle_pdf = solid_angle_pdf;
    }
}
