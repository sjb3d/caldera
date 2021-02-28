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

// limit on multi-layer BRDFs to avoid divide by zero
#define MIN_LAYER_PROBABILITY   .01f

// limit on BSDF incoding direction sample to avoid divide by zero
#define MIN_SAMPLED_N_DOT_L     1.0e-7f

/*
    Parameters for a BSDF implementation (assumes that the type
    of BSDF is already known, stored separately).

    x:  reflectance.xy (2xf16)
    y:  [31] = is_front_hit
        [30:24] = material_index
        [23:16] = roughness (u8)
        [15:0] = reflectance.z (f16)
*/
struct BsdfParams {
    uvec2 bits;
};

#define BSDF_PARAMS_Y_FRONT_HIT_BIT         0x80000000U
#define BSDF_PARAMS_Y_MATERIAL_INDEX_MASK   0x0f000000U
#define BSDF_PARAMS_Y_ROUGHNESS_MASK        0x00ff0000U
#define BSDF_PARAMS_Y_REFLECTANCE_Z_MASK    0x0000ffffU

BsdfParams create_bsdf_params(
    vec3 reflectance,
    float roughness,
    uint material_index,
    bool is_front_hit)
{
    BsdfParams p;
    p.bits.x = packHalf2x16(reflectance.xy);
    p.bits.y
        = packHalf2x16(vec2(reflectance.z, 0.f))
        | packUnorm4x8(vec4(0.f, 0.f, roughness, 0.f))
        | ((material_index << 24) & BSDF_PARAMS_Y_MATERIAL_INDEX_MASK)
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
    return unpackUnorm4x8(p.bits.y).z;
}
uint get_material_index(BsdfParams p)
{
    return (p.bits.y & BSDF_PARAMS_Y_MATERIAL_INDEX_MASK) >> 24;
}
bool is_front_hit(BsdfParams p)
{
    return (p.bits.y & BSDF_PARAMS_Y_FRONT_HIT_BIT) != 0;
}

BsdfParams replace_reflectance(BsdfParams p, vec3 reflectance)
{
    p.bits.x = packHalf2x16(reflectance.xy);
    p.bits.y &= ~BSDF_PARAMS_Y_REFLECTANCE_Z_MASK;
    p.bits.y |= packHalf2x16(vec2(reflectance.z, 0.f));
    return p;
}
BsdfParams replace_roughness(BsdfParams p, float roughness)
{
    p.bits.y &= ~BSDF_PARAMS_Y_ROUGHNESS_MASK;
    p.bits.y |= packUnorm4x8(vec4(0.f, 0.f, roughness, 0.f));
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

vec2 ggx_alpha_from_roughness(float roughness)
{
    return vec2(square(roughness));
}

#endif
