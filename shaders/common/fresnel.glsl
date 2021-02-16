#ifndef INCLUDED_COMMON_FRESNEL
#define INCLUDED_COMMON_FRESNEL

#include "maths.glsl"

// reference: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/ and PBRT book
vec3 fresnel_conductor(vec3 eta, vec3 k, float cos_theta)
{
    const float cos_theta2 = cos_theta*cos_theta;
    const vec3 two_eta_cos_theta = 2.f*eta*cos_theta;

    const vec3 t0 = eta*eta + k*k;
    const vec3 t1 = t0 * cos_theta2;
    const vec3 Rs = (t0 - two_eta_cos_theta + cos_theta2)/(t0 + two_eta_cos_theta + cos_theta2);
    const vec3 Rp = (t1 - two_eta_cos_theta + 1.f)/(t1 + two_eta_cos_theta + 1.f);

    return .5f*(Rp + Rs);
}

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
    return mix(1.f - fresnel_schlick(f0, n_dot_v), 1.f - f0, roughness);
}

vec3 refract(vec3 v, float eta)
{
    // Snell's law: sin_theta_i = sin_theta_t * eta
    const float cos_theta_i = abs(v.z);
    const float sin2_theta_i = 1.f - cos_theta_i*cos_theta_i;
    const float sin2_theta_t = sin2_theta_i/square(eta);
    const float cos_theta_t = sqrt(max(1.f - sin2_theta_t, 0.f));
    return normalize(vec3(0.f, 0.f, cos_theta_i/eta - cos_theta_t) - v/eta);
}

#endif
