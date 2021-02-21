#include "bsdf_common.glsl"

void mirror_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    in_dir = vec3(-out_dir.xy, out_dir.z);
    estimator = get_reflectance(params);
    solid_angle_pdf_or_negative = -1.f;
}
