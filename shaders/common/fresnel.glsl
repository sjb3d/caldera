#ifndef INCLUDED_COMMON_FRESNEL
#define INCLUDED_COMMON_FRESNEL

#include "maths.glsl"

// reference: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/ and PBRT book
#define FRESNEL_CONDUCTOR_IMPL(VEC_T)                                                                   \
    VEC_T fresnel_conductor(VEC_T eta, VEC_T k, float cos_theta)                                        \
    {                                                                                                   \
        const float cos_theta2 = cos_theta*cos_theta;                                                   \
        const VEC_T two_eta_cos_theta = 2.f*eta*cos_theta;                                              \
                                                                                                        \
        const VEC_T t0 = eta*eta + k*k;                                                                 \
        const VEC_T t1 = t0 * cos_theta2;                                                               \
        const VEC_T Rs = (t0 - two_eta_cos_theta + cos_theta2)/(t0 + two_eta_cos_theta + cos_theta2);   \
        const VEC_T Rp = (t1 - two_eta_cos_theta + 1.f)/(t1 + two_eta_cos_theta + 1.f);                 \
                                                                                                        \
        return .5f*(Rp + Rs);                                                                           \
    }

FRESNEL_CONDUCTOR_IMPL(float)
FRESNEL_CONDUCTOR_IMPL(vec2)
FRESNEL_CONDUCTOR_IMPL(vec3)
FRESNEL_CONDUCTOR_IMPL(vec4)

// reference: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float fresnel_dieletric(float eta, float cos_theta)
{
    const float c = cos_theta;
    const float temp = eta*eta + c*c - 1.f;
    if (temp < 0.f) {
        return 1.f;
    }

    const float g = sqrt(temp);
    return .5f
        *square((g - c)/(g + c))
        *(1.f + square(((g + c)*c - 1.f)/((g - c)*c + 1.f)));
}

// dielectric approximation
vec3 fresnel_schlick(vec3 r0, float cos_theta)
{
    return r0 + (vec3(1.f) - r0)*pow(1.f - abs(cos_theta), 5.f);
}
float fresnel_schlick(float r0, float cos_theta)
{
    return r0 + (1.f - r0)*pow(1.f - abs(cos_theta), 5.f);
}

// approximation from http://c0de517e.blogspot.com/2019/08/misunderstanding-multilayering-diffuse.html
float remaining_diffuse_strength(float n_dot_v, float f0, float roughness)
{
    const float smooth_remain = max(0.f, 1.f - fresnel_schlick(f0, n_dot_v));
    const float rough_remain = 1.f - f0;
    return mix(smooth_remain, rough_remain, roughness);
}

vec3 refract_clamp(vec3 v, vec3 n, float eta)
{
    const float n_dot_v = abs(dot(n, v));
    const float k = eta*eta - (1.f - n_dot_v*n_dot_v);
    return normalize((n_dot_v - sqrt(max(k, 0.f)))*n - v);
}

#endif
