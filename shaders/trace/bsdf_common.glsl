#ifndef INCLUDED_COMMON_BSDF
#define INCLUDED_COMMON_BSDF

#include "fresnel.glsl"

#define PLASTIC_F0              0.04f

// approximation from http://c0de517e.blogspot.com/2019/08/misunderstanding-multilayering-diffuse.html
float remaining_diffuse_strength(float n_dot_v, float roughness)
{
    return mix(1.f - fresnel_schlick(PLASTIC_F0, n_dot_v), 1.f - PLASTIC_F0, roughness);
}

/*
    Parameters for a BSDF implementation (assumes that the type
    of BSDF is already known, stored separately).
*/
struct BsdfParams {
    uvec2 bits;
};

BsdfParams create_bsdf_params(
    vec3 reflectance,
    float roughness)
{
    BsdfParams p;
    p.bits.x = packHalf2x16(reflectance.xy);
    p.bits.y = packHalf2x16(vec2(reflectance.z, roughness));
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
    return unpackHalf2x16(p.bits.y).y;
}

BsdfParams replace_roughness(BsdfParams p, float roughness)
{
    p.bits.y &= 0x0000ffff;
    p.bits.y |= packHalf2x16(vec2(0.f, roughness));
    return p;
}

#define CALLABLE_SHADER_COUNT_PER_BSDF_TYPE     2

struct BsdfEvalData {
    vec3 a;         // in: out_dir              out: f
    vec3 b;         // in: in_dir               out: (solid_angle_pdf, -, -)
    BsdfParams c;   // in: BsdfParams
};

vec3 get_in_dir(BsdfEvalData d)             { return d.a; }
vec3 get_out_dir(BsdfEvalData d)            { return d.b; }
BsdfParams get_bsdf_params(BsdfEvalData d)  { return d.c; }

BsdfEvalData write_bsdf_eval_outputs(
    vec3 f,
    float solid_angle_pdf)
{
    BsdfEvalData d;
    d.a = f;
    d.b.x = solid_angle_pdf;
    return d;
}

#define BSDF_EVAL_SHADER_INDEX(BSDF_TYPE)       ((0*CALLABLE_SHADER_COUNT_PER_BSDF_TYPE) + 0)
#define BSDF_EVAL_CALLABLE_INDEX                2
#define BSDF_EVAL_DATA(NAME)                    layout(location = 2) callableDataEXT BsdfEvalData NAME
#define BSDF_EVAL_DATA_IN(NAME)                 layout(location = 2) callableDataInEXT BsdfEvalData NAME

struct BsdfSampleData {
    vec3 a;         // in: out_dir          out: in_dir
    vec3 b;         // in: rand_u01, -      out: estimator
    BsdfParams c;   // in: BsdfParams       out: (solid_angle_pdf_or_negative, roughness_acc)
};

vec3 get_out_dir(BsdfSampleData d)              { return d.a; }
vec2 get_rand_u01(BsdfSampleData d)             { return d.b.xy; }
BsdfParams get_bsdf_params(BsdfSampleData d)    { return d.c; }

BsdfSampleData write_bsdf_sample_outputs(
    vec3 in_dir,
    vec3 estimator,
    float solid_angle_pdf_or_negative,
    float roughness_acc)
{
    BsdfSampleData d;
    d.a = in_dir;
    d.b = estimator;
    d.c.bits.x = floatBitsToUint(solid_angle_pdf_or_negative);
    d.c.bits.y = floatBitsToUint(roughness_acc);
    return d;
}

#define BSDF_SAMPLE_SHADER_INDEX(BSDF_TYPE)     ((0*CALLABLE_SHADER_COUNT_PER_BSDF_TYPE) + 1)
#define BSDF_SAMPLE_CALLABLE_INDEX              3
#define BSDF_SAMPLE_DATA(NAME)                  layout(location = 3) callableDataEXT BsdfSampleData NAME
#define BSDF_SAMPLE_DATA_IN(NAME)               layout(location = 3) callableDataInEXT BsdfSampleData NAME

#endif
