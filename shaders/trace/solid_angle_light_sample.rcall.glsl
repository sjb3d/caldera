#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

SOLID_ANGLE_LIGHT_RECORD(g_record);

LIGHT_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec2 rand_u01 = get_rand_u01(g_sample);

    const float cos_theta = 1.f - g_record.solid_angle/(2.f*PI);
    const vec3 ray_dir_ls = sample_solid_angle_uniform(cos_theta, rand_u01);

    const mat3 basis = basis_from_z_axis(normalize(g_record.direction_ws));
    const vec3 external_dir = normalize(basis*ray_dir_ls);

    g_sample = write_light_sample_outputs(
        external_dir,
        -external_dir,
        g_record.emission,
        1.f/g_record.solid_angle,
        true,
        0.f);
}
