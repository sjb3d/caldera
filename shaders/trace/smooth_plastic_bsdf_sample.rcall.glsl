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
    const float roughness = 0.f;

    const float n_dot_v = out_dir.z;
    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, roughness);
    const float spec_strength = 1.f - diffuse_strength;

    const bool sample_diffuse = split_random_variable(diffuse_strength, rand_u01.x);
    vec3 in_dir;
    float roughness_acc = 0.f;
    if (sample_diffuse) {
        in_dir = sample_hemisphere_cosine_weighted(rand_u01);
        roughness_acc = 1.f;
    } else {
        in_dir = vec3(-out_dir.xy, out_dir.z);
    }
    const float n_dot_l = in_dir.z;

    vec3 estimator;
    float solid_angle_pdf;
    if (sample_diffuse) {
        const vec3 diff_f = reflectance*(diffuse_strength/PI);

        estimator = diff_f/(diffuse_strength*get_hemisphere_cosine_weighted_proj_pdf());
        solid_angle_pdf = diffuse_strength*get_hemisphere_cosine_weighted_pdf(n_dot_l);
    } else {
        const float spec_f = fresnel_schlick(PLASTIC_F0, n_dot_v);

        estimator = vec3(spec_f)/spec_strength;
        solid_angle_pdf = -1.f;
    }

    g_sample = write_bsdf_sample_outputs(in_dir, estimator, solid_angle_pdf, roughness_acc);
}
