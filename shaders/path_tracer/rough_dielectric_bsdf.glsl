#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "fresnel.glsl"
#include "ggx.glsl"

void rough_dielectric_bsdf_eval(
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out HERO_VEC f,
    out float solid_angle_pdf)
{
    const HERO_VEC reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = ggx_alpha_from_roughness(roughness);

    const vec3 v = out_dir;
    const vec3 l = in_dir;
    const float n_dot_l = in_dir.z;
    const float n_dot_v = out_dir.z;

    const float eta = is_front_hit(params) ? DIELECTRIC_IOR : (1.f/DIELECTRIC_IOR);
    const float reflect_probability = fresnel_dieletric(eta, out_dir.z);

    vec3 h;
    float selection_probability;
    if (sign_bit_set(n_dot_l)) {
        h = normalize(v + l*eta);
        selection_probability = 1.f - reflect_probability;
    } else {
        h = normalize(v + l);
        selection_probability = reflect_probability;
    }

    f   = reflectance
        * selection_probability
        * ggx_brdf_without_fresnel(h, v, l, alpha);

    solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha) * selection_probability;
}

void rough_dielectric_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec3 bsdf_rand_u01,
    out vec3 in_dir,
    out HERO_VEC estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const HERO_VEC reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = ggx_alpha_from_roughness(roughness);

    const vec3 v = out_dir;
    const vec3 h = sample_vndf(out_dir, alpha, bsdf_rand_u01.xy);

    const float eta = is_front_hit(params) ? DIELECTRIC_IOR : (1.f/DIELECTRIC_IOR);
    const float reflect_probability = fresnel_dieletric(eta, out_dir.z);

    float selection_probability;
    if (bsdf_rand_u01.z > reflect_probability) {
        in_dir = refract_clamp(out_dir, h, eta);
        selection_probability = reflect_probability;
    } else {
        in_dir = reflect(-out_dir, h);
        in_dir.z = max(in_dir.z, MIN_SAMPLED_N_DOT_L);
        selection_probability = 1.f - reflect_probability;
    }

    const vec3 l = in_dir;
    estimator
        = reflectance
        * ggx_vndf_sampled_estimator_without_fresnel(v, l, alpha);

    solid_angle_pdf_or_negative = ggx_vndf_sampled_pdf(v, h, alpha) * selection_probability;
}
