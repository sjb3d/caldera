#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "fresnel.glsl"

void smooth_plastic_bsdf_eval(
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vec3 f,
    out float solid_angle_pdf)
{
    const vec3 reflectance = get_reflectance(params);
    const float roughness = 0.f;

    const float n_dot_l = in_dir.z;
    const float n_dot_v = out_dir.z;
    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, PLASTIC_F0, roughness);

    const vec3 diff_f = reflectance*(diffuse_strength/PI);
    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);

    f = diff_f;
    solid_angle_pdf = diffuse_strength*diff_solid_angle_pdf;
}

void smooth_plastic_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    out float sampled_roughness)
{
    const vec3 reflectance = get_reflectance(params);
    const float roughness = 0.f;

    const float n_dot_v = out_dir.z;
    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, PLASTIC_F0, roughness);
    const float spec_strength = 1.f - diffuse_strength;

    const bool sample_diffuse = split_random_variable(diffuse_strength, rand_u01.x);
    if (sample_diffuse) {
        in_dir = sample_hemisphere_cosine_weighted(rand_u01);
        sampled_roughness = 1.f;
    } else {
        in_dir = vec3(-out_dir.xy, out_dir.z);
        sampled_roughness = roughness;
    }
    const float n_dot_l = in_dir.z;

    if (sample_diffuse) {
        const vec3 diff_f = reflectance*(diffuse_strength/PI);

        estimator = diff_f/(diffuse_strength*get_hemisphere_cosine_weighted_proj_pdf());
        solid_angle_pdf_or_negative = diffuse_strength*get_hemisphere_cosine_weighted_pdf(n_dot_l);
    } else {
        const float spec_f = fresnel_schlick(PLASTIC_F0, n_dot_v);

        estimator = vec3(spec_f)/spec_strength;
        solid_angle_pdf_or_negative = -1.f;
    }
}
