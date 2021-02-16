#ifndef INCLUDED_COMMON_BSDF
#define INCLUDED_COMMON_BSDF

#include "maths.glsl"
#include "sampler.glsl"

#define PLASTIC_F0              0.04f

#define DIELECTRIC_IOR          1.333f

// vaguely Aluminium, need to convert some spectral data
#define CONDUCTOR_ETA           vec3(1.09f)
#define CONDUCTOR_K             vec3(6.79f)

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

#define BSDF_TYPE_DIFFUSE           0
#define BSDF_TYPE_MIRROR            1
#define BSDF_TYPE_SMOOTH_DIELECTRIC 2
#define BSDF_TYPE_SMOOTH_PLASTIC    3
#define BSDF_TYPE_ROUGH_PLASTIC     4
#define BSDF_TYPE_ROUGH_CONDUCTOR   5
#define BSDF_TYPE_COUNT             6

bool bsdf_is_always_delta(uint bsdf_type)
{
    return (bsdf_type == BSDF_TYPE_MIRROR || bsdf_type == BSDF_TYPE_SMOOTH_DIELECTRIC);
}
bool bsdf_has_transmission(uint bsdf_type)
{
    return (bsdf_type == BSDF_TYPE_SMOOTH_DIELECTRIC);
}

#define CALLABLE_SHADER_COUNT_PER_BSDF_TYPE     2

struct BsdfEvalData {
    vec3 a;         // in: out_dir              out: f
    vec3 b;         // in: in_dir               out: (solid_angle_pdf, -, -)
    BsdfParams c;   // in: BsdfParams
};

vec3 get_out_dir(BsdfEvalData d)            { return d.a; }
vec3 get_in_dir(BsdfEvalData d)             { return d.b; }
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

#define BSDF_EVAL_SHADER_INDEX(BSDF_TYPE)       ((BSDF_TYPE)*CALLABLE_SHADER_COUNT_PER_BSDF_TYPE + 0)
#define BSDF_EVAL_CALLABLE_INDEX                2
#define BSDF_EVAL_DATA(NAME)                    layout(location = BSDF_EVAL_CALLABLE_INDEX) callableDataEXT BsdfEvalData NAME
#define BSDF_EVAL_DATA_IN(NAME)                 layout(location = BSDF_EVAL_CALLABLE_INDEX) callableDataInEXT BsdfEvalData NAME

struct BsdfSampleData {
    vec3 a;         // in: out_dir          out: in_dir
    vec3 b;         // in: rand_u01, -      out: estimator
    BsdfParams c;   // in: BsdfParams       out: (solid_angle_pdf_or_negative, sampled_roughness)
};

vec3 get_out_dir(BsdfSampleData d)              { return d.a; }
vec2 get_rand_u01(BsdfSampleData d)             { return d.b.xy; }
BsdfParams get_bsdf_params(BsdfSampleData d)    { return d.c; }

BsdfSampleData write_bsdf_sample_outputs(
    vec3 in_dir,
    vec3 estimator,
    float solid_angle_pdf_or_negative,
    float sampled_roughness)
{
    BsdfSampleData d;
    d.a = in_dir;
    d.b = estimator;
    d.c.bits.x = floatBitsToUint(solid_angle_pdf_or_negative);
    d.c.bits.y = floatBitsToUint(sampled_roughness);
    return d;
}

#define BSDF_SAMPLE_SHADER_INDEX(BSDF_TYPE)     ((BSDF_TYPE)*CALLABLE_SHADER_COUNT_PER_BSDF_TYPE + 1)
#define BSDF_SAMPLE_CALLABLE_INDEX              3
#define BSDF_SAMPLE_DATA(NAME)                  layout(location = BSDF_SAMPLE_CALLABLE_INDEX) callableDataEXT BsdfSampleData NAME
#define BSDF_SAMPLE_DATA_IN(NAME)               layout(location = BSDF_SAMPLE_CALLABLE_INDEX) callableDataInEXT BsdfSampleData NAME

#define BSDF_EVAL_MAIN(FUNC)                                    \
    BSDF_EVAL_DATA_IN(g_eval);                                  \
    void main() {                                               \
        vec3 f;                                                 \
        float solid_angle_pdf;                                  \
        FUNC(                                                   \
            get_out_dir(g_eval),                                \
            get_in_dir(g_eval),                                 \
            get_bsdf_params(g_eval),                            \
            f,                                                  \
            solid_angle_pdf);                                   \
        g_eval = write_bsdf_eval_outputs(f, solid_angle_pdf);   \
    }

#define BSDF_SAMPLE_MAIN(FUNC)                                  \
    BSDF_SAMPLE_DATA_IN(g_sample);                              \
    void main() {                                               \
        vec3 in_dir;                                            \
        vec3 estimator;                                         \
        float solid_angle_pdf_or_negative;                      \
        float sampled_roughness;                                \
        FUNC(                                                   \
            get_out_dir(g_sample),                              \
            get_bsdf_params(g_sample),                          \
            get_rand_u01(g_sample),                             \
            in_dir,                                             \
            estimator,                                          \
            solid_angle_pdf_or_negative,                        \
            sampled_roughness);                                 \
        g_sample = write_bsdf_sample_outputs(                   \
            in_dir,                                             \
            estimator,                                          \
            solid_angle_pdf_or_negative,                        \
            sampled_roughness);                                 \
    }

#endif
