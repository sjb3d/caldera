#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "fresnel.glsl"

void smooth_dielectric_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec3 bsdf_rand_u01,
    out vec3 in_dir,
    out vecW estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const vecW reflectance = get_reflectance(params);

    const float eta = is_front_hit(params) ? DIELECTRIC_IOR : (1.f/DIELECTRIC_IOR);
    const float reflect_chance = fresnel_dieletric(eta, out_dir.z);

    if (bsdf_rand_u01.x > reflect_chance) {
        in_dir = refract_clamp(out_dir, vec3(0.f, 0.f, 1.f), eta);
    } else {
        in_dir = vec3(-out_dir.xy, out_dir.z);
    }
    estimator = reflectance;
    solid_angle_pdf_or_negative = -1.f;
}
