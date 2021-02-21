#include "bsdf_common.glsl"
#include "sampler.glsl"
#include "fresnel.glsl"

void smooth_dielectric_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const vec3 reflectance = get_reflectance(params);

    const float eta = is_front_hit(params) ? DIELECTRIC_IOR : (1.f/DIELECTRIC_IOR);
    const float reflect_chance = fresnel_dieletric(eta, out_dir.z);

    if (rand_u01.x > reflect_chance) {
        in_dir = refract(out_dir, eta);
    } else {
        in_dir = vec3(-out_dir.xy, out_dir.z);
    }
    estimator = reflectance;
    solid_angle_pdf_or_negative = -1.f;
}
