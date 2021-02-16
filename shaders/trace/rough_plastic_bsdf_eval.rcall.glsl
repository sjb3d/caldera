#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "bsdf_common.glsl"
#include "ggx.glsl"

BSDF_EVAL_DATA_IN(g_eval);

void main()
{
    const vec3 in_dir = get_in_dir(g_eval);
    const vec3 out_dir = get_out_dir(g_eval);
    const BsdfParams params = get_bsdf_params(g_eval);

    const vec3 reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = vec2(square(roughness));

    const vec3 v = out_dir;
    const vec3 l = in_dir;
    const float n_dot_l = in_dir.z;
    const float n_dot_v = out_dir.z;
    const vec3 h = normalize(v + l);
    const float h_dot_v = abs(dot(h, v));

    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, roughness);

    const vec3 diff_f = reflectance*(diffuse_strength/PI);
    const float spec_f = fresnel_schlick(PLASTIC_F0, h_dot_v)*ggx_brdf_without_fresnel(h, v, l, alpha);

    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
    const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
    const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_strength);

    g_eval = write_bsdf_eval_outputs(diff_f + vec3(spec_f), combined_solid_angle_pdf);
}
