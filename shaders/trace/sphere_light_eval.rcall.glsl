#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

SPHERE_LIGHT_RECORD(g_light);

LIGHT_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 light_position = g_eval.position;
    const vec3 target_position = g_eval.emission;
    //const vec3 target_normal = g_eval.normal;

    const vec3 light_normal = normalize(light_position - g_light.centre_ws);
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    const vec3 emission = (facing_term > 0.f) ? g_light.emission : vec3(0.f);
    
    const float distance_sq = dot(target_from_light, target_from_light);
    const float area_ws = 4.f * PI * g_light.radius_ws * g_light.radius_ws;
    const float solid_angle_pdf = solid_angle_pdf_from_area_pdf(1.f/area_ws, facing_term, distance_sq);

    g_eval.normal = light_normal;
    g_eval.emission = emission;
    g_eval.solid_angle_pdf = solid_angle_pdf;
}
