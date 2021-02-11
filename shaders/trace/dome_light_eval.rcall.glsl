#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

DOME_LIGHT_RECORD(g_record);

LIGHT_EVAL_DATA_IN(g_eval);

void main()
{
    //const vec3 light_position = g_eval.position;
    //const vec3 target_position = g_eval.emission;

    const float solid_angle_pdf = 1.f/(4.f*PI);

    g_eval.emission = g_record.emission;
    g_eval.solid_angle_pdf = solid_angle_pdf;
}
