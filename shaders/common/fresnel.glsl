#ifndef INCLUDED_COMMON_FRESNEL
#define INCLUDED_COMMON_FRESNEL

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
    cos_theta = abs(cos_theta);
    return r0 + (vec3(1.f) - r0)*pow(1.f - cos_theta, 5.f);
}
float fresnel_schlick(float r0, float cos_theta)
{
    cos_theta = abs(cos_theta);
    return r0 + (1.f - r0)*pow(1.f - cos_theta, 5.f);
}

#endif
