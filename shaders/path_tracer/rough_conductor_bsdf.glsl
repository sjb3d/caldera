#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "ggx.glsl"
#include "fresnel.glsl"

void get_eta_k(
    sampler1DArray conductors,
    vecW wavelengths,
    uint material_index,
    out vecW eta,
    out vecW k)
{
    const vecW wavelengths_u = unlerp(
        vecW(SMITS_WAVELENGTH_MIN),
        vecW(SMITS_WAVELENGTH_MAX),
        wavelengths);
    const float layer = float(material_index);
    const vec2 eta_k0 = texture(conductors, vec2(wavelengths_u.x, layer)).xy;
    const vec2 eta_k1 = texture(conductors, vec2(wavelengths_u.y, layer)).xy;
#if WAVELENGTHS_PER_RAY >= 3
    const vec2 eta_k2 = texture(conductors, vec2(wavelengths_u.z, layer)).xy;
#endif
#if WAVELENGTHS_PER_RAY >= 4
    const vec2 eta_k3 = texture(conductors, vec2(wavelengths_u.w, layer)).xy;
#endif
    eta = PER_WAVELENGTH(eta_k0.x, eta_k1.x, eta_k2.x, eta_k3.x);
    k = PER_WAVELENGTH(eta_k0.y, eta_k1.y, eta_k2.y, eta_k3.y);
}

void rough_conductor_bsdf_eval(
    sampler1DArray conductors,
    vecW wavelengths,
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vecW f,
    out float solid_angle_pdf)
{
    const vecW reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = ggx_alpha_from_roughness(roughness);

    const vec3 v = out_dir;
    const vec3 l = in_dir;
    const float n_dot_l = in_dir.z;
    const float n_dot_v = out_dir.z;
    const vec3 h = normalize(v + l);
    const float h_dot_v = abs(dot(h, v));

    vecW eta, k;
    get_eta_k(conductors, wavelengths, get_material_index(params), eta, k);

    f   = reflectance
        * fresnel_conductor(eta, k, h_dot_v)
        * ggx_brdf_without_fresnel(h, v, l, alpha);

    solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);
}

void rough_conductor_bsdf_sample(
    sampler1DArray conductors,
    vecW wavelengths,
    vec3 out_dir,
    BsdfParams params,
    vec3 bsdf_rand_u01,
    out vec3 in_dir,
    out vecW estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const vecW reflectance = get_reflectance(params);
    const float roughness = get_roughness(params);
    const vec2 alpha = ggx_alpha_from_roughness(roughness);

    const vec3 v = out_dir;

    const vec3 h = sample_vndf(out_dir, alpha, bsdf_rand_u01.xy);
    in_dir = reflect(-out_dir, h);
    in_dir.z = max(in_dir.z, MIN_SAMPLED_N_DOT_L);

    const vec3 l = in_dir;
    const float h_dot_v = abs(dot(h, v));

    vecW eta, k;
    get_eta_k(conductors, wavelengths, get_material_index(params), eta, k);

    estimator
        = reflectance
        * fresnel_conductor(eta, k, h_dot_v)
        * ggx_vndf_sampled_estimator_without_fresnel(v, l, alpha);

    solid_angle_pdf_or_negative = ggx_vndf_sampled_pdf(v, h, alpha);
}
