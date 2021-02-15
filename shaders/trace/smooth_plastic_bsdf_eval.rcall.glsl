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

    const vec3 reflectance = get_reflectance(params);

    const float diffuse_strength = remaining_diffuse_strength(out_dir.z, 0.f);

    const vec3 diff_f = reflectance*(diffuse_strength/PI);
    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(in_dir.z);

    g_eval = write_bsdf_eval_outputs(diff_f, diffuse_strength*diff_solid_angle_pdf);
}
