#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

QUAD_LIGHT_RECORD(g_light);

LIGHT_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec3 target_position = g_sample.position;
    //const vec3 target_normal = g_sample.normal;
    const vec2 rand_u01 = g_sample.emission.xy;

    const vec3 light_position = g_light.corner_ws + rand_u01.x*g_light.edge0_ws + rand_u01.y*g_light.edge1_ws;
    const vec3 light_normal = g_light.normal_ws;
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    const vec3 emission = (facing_term > 0.f) ? g_light.emission : vec3(0.f);

    const float distance_sq = dot(target_from_light, target_from_light);
    const float solid_angle_pdf = solid_angle_pdf_from_area_pdf(g_light.area_pdf, facing_term, distance_sq);

    g_sample.position = light_position;
    g_sample.normal = light_normal;
    g_sample.emission = emission;
    g_sample.solid_angle_pdf = solid_angle_pdf;
    g_sample.unit_value = g_light.unit_value;   
}
