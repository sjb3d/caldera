#include "bsdf_common.glsl"
#include "sampler.glsl"

void diffuse_bsdf_eval(
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vecW f,
    out float solid_angle_pdf)
{
    const float n_dot_l = in_dir.z; 
    const vecW reflectance = get_reflectance(params);

    f = reflectance/PI;
    solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
}

void diffuse_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec3 bsdf_rand_u01,
    out vec3 in_dir,
    out vecW estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const vecW reflectance = get_reflectance(params);
    
    in_dir = sample_hemisphere_cosine_weighted(bsdf_rand_u01.xy);
    in_dir.z = max(in_dir.z, MIN_SAMPLED_N_DOT_L);

    const float n_dot_l = in_dir.z;

    estimator = reflectance;
    solid_angle_pdf_or_negative = get_hemisphere_cosine_weighted_pdf(n_dot_l);   
}
