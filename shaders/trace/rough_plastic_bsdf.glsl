#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "ggx.glsl"
#include "fresnel.glsl"

void rough_plastic_bsdf_eval(
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vec3 f,
    out float solid_angle_pdf)
{
    const vec3 reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = vec2(square(roughness));

    const vec3 v = out_dir;
    const vec3 l = in_dir;
    const float n_dot_l = in_dir.z;
    const float n_dot_v = out_dir.z;
    const vec3 h = normalize(v + l);
    const float h_dot_v = abs(dot(h, v));

    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, PLASTIC_F0, roughness);
    const float diffuse_probability = max(diffuse_strength, MIN_LAYER_PROBABILITY);

    const vec3 diff_f = reflectance*(diffuse_strength/PI);
    const float spec_f = fresnel_schlick(PLASTIC_F0, h_dot_v)*ggx_brdf_without_fresnel(h, v, l, alpha);

    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
    const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
    const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_probability);

    f = diff_f + vec3(spec_f);
    solid_angle_pdf = combined_solid_angle_pdf;
}

void rough_plastic_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    out float sampled_roughness)
{
    const vec3 reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = vec2(square(roughness));

    const vec3 v = out_dir;
    const float n_dot_v = out_dir.z;
    const float diffuse_strength = remaining_diffuse_strength(n_dot_v, PLASTIC_F0, roughness);
    const float diffuse_probability = max(diffuse_strength, MIN_LAYER_PROBABILITY);

    const bool sample_diffuse = split_random_variable(diffuse_probability, rand_u01.x);
    if (sample_diffuse) {
        in_dir = sample_hemisphere_cosine_weighted(rand_u01);
        sampled_roughness = 1.f;
    } else {
        const vec3 h = sample_vndf(out_dir, alpha, rand_u01);
        in_dir = reflect(-out_dir, h);
        sampled_roughness = roughness;
    }
    in_dir.z = max(in_dir.z, MIN_SAMPLED_N_DOT_L);

    const vec3 l = in_dir;
    const vec3 h = normalize(v + l);
    const float h_dot_v = abs(dot(h, v));
    const float n_dot_l = l.z;

    const float diff_solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
    const float spec_solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
    const float combined_solid_angle_pdf = mix(spec_solid_angle_pdf, diff_solid_angle_pdf, diffuse_probability);

    const float spec_f = fresnel_schlick(PLASTIC_F0, h_dot_v)*ggx_brdf_without_fresnel(h, v, l, alpha);
    const vec3 diff_f = reflectance*(diffuse_strength/PI);

    const vec3 f = vec3(spec_f) + diff_f;
    estimator = f * n_dot_l / combined_solid_angle_pdf;
    solid_angle_pdf_or_negative = combined_solid_angle_pdf;
}
