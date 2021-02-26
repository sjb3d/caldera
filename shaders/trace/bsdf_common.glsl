#ifndef INCLUDED_COMMON_BSDF
#define INCLUDED_COMMON_BSDF

#include "maths.glsl"
#include "sampler.glsl"

// fixed for now
// TODO: choose from palette of dieletric materials
#define PLASTIC_F0              0.04f

// fixed for now
// TODO: choose from palette of dieletric materials
#define DIELECTRIC_IOR          1.333f

// vaguely Aluminium, need to convert some spectral data
// TODO: choose from palette of conductor materials
#define CONDUCTOR_ETA           1.09f
#define CONDUCTOR_K             6.79f

// limit on multi-layer BRDFs to avoid divide by zero
#define MIN_LAYER_PROBABILITY   .01f

// limit on BSDF incoding direction sample to avoid divide by zero
#define MIN_SAMPLED_N_DOT_L     1.0e-7f

/*
    Parameters for a BSDF implementation (assumes that the type
    of BSDF is already known, stored separately).
*/
struct BsdfParams {
    uvec2 bits;
};

#define BSDF_PARAMS_Y_FRONT_HIT_BIT     0x80000000U

BsdfParams create_bsdf_params(
    vec3 reflectance,
    float roughness,
    bool is_front_hit)
{
    BsdfParams p;
    p.bits.x = packHalf2x16(reflectance.xy);
    p.bits.y
        = packHalf2x16(vec2(reflectance.z, abs(roughness)))
        | (is_front_hit ? BSDF_PARAMS_Y_FRONT_HIT_BIT : 0)

        ;
    return p;
}

vec3 get_reflectance(BsdfParams p)
{
    return vec3(
        unpackHalf2x16(p.bits.x),
        unpackHalf2x16(p.bits.y).x);
}
float get_roughness(BsdfParams p)
{
    return abs(unpackHalf2x16(p.bits.y).y);
}
bool is_front_hit(BsdfParams p)
{
    return (p.bits.y & BSDF_PARAMS_Y_FRONT_HIT_BIT) != 0;
}

BsdfParams replace_reflectance(BsdfParams p, vec3 reflectance)
{
    p.bits.x = packHalf2x16(reflectance.xy);
    p.bits.y &= 0xffff0000;
    p.bits.y |= packHalf2x16(vec2(reflectance.z, 0.f));
    return p;
}
BsdfParams replace_roughness(BsdfParams p, float roughness)
{
    p.bits.y &= (BSDF_PARAMS_Y_FRONT_HIT_BIT | 0x0000ffff);
    p.bits.y |= packHalf2x16(vec2(0.f, abs(roughness)));
    return p;
}

#define BSDF_TYPE_NONE              0
#define BSDF_TYPE_DIFFUSE           1
#define BSDF_TYPE_MIRROR            2
#define BSDF_TYPE_SMOOTH_DIELECTRIC 3
#define BSDF_TYPE_ROUGH_DIELECTRIC  4
#define BSDF_TYPE_SMOOTH_PLASTIC    5
#define BSDF_TYPE_ROUGH_PLASTIC     6
#define BSDF_TYPE_ROUGH_CONDUCTOR   7

bool bsdf_is_always_delta(uint bsdf_type)
{
    return (bsdf_type == BSDF_TYPE_MIRROR || bsdf_type == BSDF_TYPE_SMOOTH_DIELECTRIC);
}
bool bsdf_has_transmission(uint bsdf_type)
{
    return (bsdf_type == BSDF_TYPE_SMOOTH_DIELECTRIC) || (bsdf_type == BSDF_TYPE_ROUGH_DIELECTRIC);
}

#endif
