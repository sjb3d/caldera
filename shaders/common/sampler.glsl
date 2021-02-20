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

uint get_seq_hash(uvec2 pixel_coord, uint seq_index, uint sample_index)
{
    return hash((seq_index << 24) ^ (pixel_coord.y << 12) ^ pixel_coord.x);
}

bool split_random_variable(float accept_probability, inout float u01)
{
    const bool is_accept = (u01 <= accept_probability);
    if (is_accept) {
        u01 /= accept_probability;
    } else {
        u01 -= accept_probability;
        u01 /= (1.f - accept_probability);
    }
    return is_accept;
}

float solid_angle_pdf_from_area_pdf(float area_pdf, float cos_theta, float distance_sq)
{
    return area_pdf * distance_sq / abs(cos_theta);
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
    return r*vec2(cos(phi), sin(phi));
}

vec3 sample_hemisphere_cosine_weighted(vec2 u)
{
    const vec2 disc_pos = sample_disc_uniform(u);
    const float z = sqrt(max(1.f - dot(disc_pos, disc_pos), 0.f));
    return vec3(disc_pos, z);
}

float get_hemisphere_cosine_weighted_pdf(float cos_theta)
{
    return abs(cos_theta)/PI;
}

float get_hemisphere_cosine_weighted_proj_pdf()
{
    return 1.f/PI;
}

vec3 dir_from_phi_cos_theta(float phi, float cos_theta)
{
    const float sin_theta = sqrt(max(0.f, 1.f - cos_theta*cos_theta));
    return vec3(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
}

vec3 sample_sphere_uniform(vec2 u)
{
    const float cos_theta = 2.f*u.x - 1.f;
    const float phi = 2.f*PI*u.y;
    return dir_from_phi_cos_theta(phi, cos_theta);
}

vec3 sample_solid_angle_uniform(float cos_theta_min, vec2 u)
{
    const float cos_theta = 1.f + (cos_theta_min - 1.f)*u.x;
    const float phi = 2.f*PI*u.y;
    return dir_from_phi_cos_theta(phi, cos_theta);
}

#endif
