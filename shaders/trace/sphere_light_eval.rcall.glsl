#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "light_common.glsl"

SPHERE_LIGHT_RECORD(g_light);

LIGHT_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 light_position = g_eval.position;
    const vec3 target_position = g_eval.emission;
    const vec3 target_normal = g_eval.normal;

    const vec3 light_normal = normalize(light_position - g_light.centre_ws);
    const vec3 target_from_light = target_position - light_position;
    const float light_facing_term = dot(target_from_light, light_normal);
    const vec3 light_emission = (light_facing_term > 0.f) ? g_light.emission : vec3(0.f);
    const float light_area_pdf = g_light.area_pdf;

    g_eval.normal = light_normal;
    g_eval.emission = light_emission;
    g_eval.area_pdf = light_area_pdf;
}
