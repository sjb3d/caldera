#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "bsdf_common.glsl"
#include "ggx.glsl"

BSDF_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec3 out_dir = get_out_dir(g_sample);
    const BsdfParams params = get_bsdf_params(g_sample);
    vec2 rand_u01 = get_rand_u01(g_sample);

    const vec3 reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = vec2(square(roughness));

    const vec3 v = out_dir;
    const float n_dot_v = out_dir.z;
    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, roughness);

    const bool sample_diffuse = split_random_variable(diffuse_strength, rand_u01.x);
    vec3 in_dir;
    if (sample_diffuse) {
        in_dir = sample_hemisphere_cosine_weighted(rand_u01);
    } else {
        const vec3 h = sample_vndf(out_dir, alpha, rand_u01);
        in_dir = reflect(-out_dir, h);
        in_dir.z = abs(in_dir.z);
    }

    const vec3 l = in_dir;
    const vec3 h = normalize(v + l);
    const float h_dot_v = abs(dot(h, v));
    const float n_dot_l = l.z;

    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
    const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
    const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_strength);

    const float spec_f = fresnel_schlick(PLASTIC_F0, h_dot_v)*ggx_brdf_without_fresnel(h, v, l, alpha);
    const vec3 diff_f = reflectance*(diffuse_strength/PI);

    const vec3 f = vec3(spec_f) + diff_f;
    const vec3 estimator = f * n_dot_l / combined_solid_angle_pdf;

    g_sample = write_bsdf_sample_outputs(in_dir, estimator, combined_solid_angle_pdf, 0.f);
}
