#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "bsdf_common.glsl"

BSDF_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec3 out_dir = get_out_dir(g_sample);
    const BsdfParams params = get_bsdf_params(g_sample);
    vec2 rand_u01 = get_rand_u01(g_sample);

    const vec3 reflectance = get_reflectance(params);
    
    const vec3 in_dir = sample_hemisphere_cosine_weighted(rand_u01);
    const float n_dot_l = in_dir.z;

    const vec3 f = reflectance/PI;
    const vec3 estimator = f/get_hemisphere_cosine_weighted_proj_pdf();
    const float solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);

    g_sample = write_bsdf_sample_outputs(in_dir, estimator, solid_angle_pdf, 0.f);
}
