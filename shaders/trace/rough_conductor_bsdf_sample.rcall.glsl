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

    const vec3 h = sample_vndf(out_dir, alpha, rand_u01);
    vec3 in_dir = reflect(-out_dir, h);
    in_dir.z = abs(in_dir.z);

    const vec3 l = in_dir;
    const float h_dot_v = abs(dot(h, v));
    const vec3 estimator
        = reflectance
        * fresnel_conductor(CONDUCTOR_ETA, CONDUCTOR_K, h_dot_v)
        * ggx_vndf_sampled_estimator_without_fresnel(v, l, alpha);

    const float solid_angle_pdf = ggx_vndf_sampled_pdf(v, h, alpha);

    g_sample = write_bsdf_sample_outputs(in_dir, estimator, solid_angle_pdf, 0.f);
}
