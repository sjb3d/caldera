#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

SPHERE_LIGHT_RECORD(g_light);

LIGHT_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec3 target_position = g_sample.position;
    //const vec3 target_normal = g_sample.normal;
    const vec2 rand_u01 = g_sample.emission.xy;

    const vec3 sample_dir = sample_sphere_uniform(rand_u01);

    const vec3 light_position = g_light.centre_ws + g_light.radius_ws*sample_dir;
    const vec3 light_normal = sample_dir;
    const vec3 target_from_light = target_position - light_position;
    const float light_facing_term = dot(target_from_light, light_normal);
    const vec3 light_emission = (light_facing_term > 0.f) ? g_light.emission : vec3(0.f);

    g_sample.position = light_position;
    g_sample.normal = light_normal;
    g_sample.emission = light_emission;
    g_sample.area_pdf = g_light.area_pdf;
    g_sample.unit_value = g_light.unit_value;   
}
