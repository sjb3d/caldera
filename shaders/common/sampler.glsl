#ifndef INCLUDED_COMMON_SAMPLER
#define INCLUDED_COMMON_SAMPLER

#include "maths.glsl"

uint hash(uint a)
{
    // see http://burtleburtle.net/bob/hash/integer.html
    a -= (a << 6);
    a ^= (a >> 17);
    a -= (a << 9);
    a ^= (a << 4);
    a -= (a << 3);
    a ^= (a << 10);
    a ^= (a >> 15);
    return a;
}

vec2 sample_disc_uniform(vec2 u)
{
    // remap to [-1,1]
    const float a = 2.0f*u.x - 1.0f;
    const float b = 2.0f*u.y - 1.0f;

    // use negative radius trick
    // http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    float r, phi;
    if (abs(a) > abs(b)) {
        r = a;
        phi = 0.25f*PI*b/a;
    } else {
        r = b;
        phi = 0.5f*PI - 0.25f*PI*a/b;
    }

    // convert to xy coordinate
    vec2 dir = vec2(0.f);
    if (!isinf(phi)) {
        dir = vec2(cos(phi), sin(phi));
    }
    return r*dir;
}

vec3 sample_hemisphere_cosine_weighted(vec2 u)
{
    const vec2 disc_pos = sample_disc_uniform(u);
    const float z = sqrt(max(1.f - dot(disc_pos, disc_pos), 0.f));
    return vec3(disc_pos, z);
}

vec3 sample_ggx_vndf(vec3 Ve, vec2 alpha, vec2 u)
{
    // transforming the view direction to the hemisphere configuration
    const vec3 Vh = normalize(vec3(alpha.x*Ve.x, alpha.y*Ve.y, Ve.z));
    // orthonormal basis
    const vec3 T1 = (Vh.z < .9999f) ? normalize(cross(vec3(0.f, 0.f, 1.f), Vh)) : vec3(1.f, 0.f, 0.f);
    const vec3 T2 = cross(Vh, T1);
    // parameterization of the projected area
    vec2 p = sample_disc_uniform(u);
    const float s = .5f*(1.f + Vh.z);
    p.y = mix(sqrt(max(1.f - p.x*p.x, 0.f)), p.y, s);
    // reprojection onto hemisphere
    const vec3 Nh = p.x*T1 + p.y*T2 + sqrt(max(0.f, 1.f - dot(p, p)))*Vh;
    // transforming the normal back to the ellipsoid configuration
    return normalize(vec3(alpha.x*Nh.x, alpha.y*Nh.y, max(0.f, Nh.z)));
}

#endif
