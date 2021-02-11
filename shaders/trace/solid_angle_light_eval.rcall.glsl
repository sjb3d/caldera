#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

LIGHT_UNIFORM_DATA(g_light);

SOLID_ANGLE_LIGHT_RECORD(g_record);

LIGHT_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 light_direction = g_eval.position_or_extdir;
    const vec3 target_position = g_eval.emission;

    // check we are within the solid angle of the light
    const float cos_theta = dot(g_record.direction_ws, light_direction);
    const float cos_theta_min = 1.f - g_record.solid_angle/(2.f*PI);
    const vec3 emission = (cos_theta > cos_theta_min) ? g_record.emission : vec3(0.f);

    g_eval.emission = emission;
    g_eval.solid_angle_pdf = 1.f/g_record.solid_angle;
}
