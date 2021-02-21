#include "bsdf_common.glsl"
#include "sampler.glsl"

void diffuse_bsdf_eval(
    vec3 out_dir,
    vec3 in_dir,
    BsdfParams params,
    out vec3 f,
    out float solid_angle_pdf)
{
    const float n_dot_l = in_dir.z; 
    const vec3 reflectance = get_reflectance(params);

    f = reflectance/PI;
    solid_angle_pdf = get_hemisphere_cosine_weighted_pdf(n_dot_l);
}

void diffuse_bsdf_sample(
    vec3 out_dir,
    BsdfParams params,
    vec2 rand_u01,
    out vec3 in_dir,
    out vec3 estimator,
    out float solid_angle_pdf_or_negative,
    inout float path_max_roughness)
{
    const vec3 reflectance = get_reflectance(params);
    
    in_dir = sample_hemisphere_cosine_weighted(rand_u01);
    in_dir.z = max(in_dir.z, MIN_SAMPLED_N_DOT_L);

    const float n_dot_l = in_dir.z;

    estimator = reflectance;
    solid_angle_pdf_or_negative = get_hemisphere_cosine_weighted_pdf(n_dot_l);   
}
