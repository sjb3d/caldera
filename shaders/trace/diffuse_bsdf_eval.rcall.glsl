#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "bsdf_common.glsl"

BSDF_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 in_dir = get_in_dir(g_eval);
    const vec3 out_dir = get_out_dir(g_eval);
    const BsdfParams params = get_bsdf_params(g_eval);

    const float n_dot_l = in_dir.z; 
    const vec3 reflectance = get_reflectance(params);

    const vec3 f = reflectance/PI;
    const float solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);

    g_eval = write_bsdf_eval_outputs(f, solid_angle_pdf);
}
