#include "normal_pack.glsl"

struct LightAliasEntry {
    float split;
    uint indices;
};

#define LIGHT_UNIFORM_DATA(NAME)                                        \
    layout(buffer_reference, scalar) buffer LightProbabilityTable {     \
        float entries[];                                                \
    };                                                                  \
    layout(buffer_reference, scalar) buffer LightAliasTable {           \
        LightAliasEntry entries[];                                      \
    };                                                                  \
    layout(set = 0, binding = 1, scalar) uniform LightUniforms {        \
        LightProbabilityTable probability_table;                        \
        LightAliasTable alias_table;                                    \
        uint sample_sphere_solid_angle;                                 \
        uint sampled_count;                                             \
        uint external_begin;                                            \
        uint external_end;                                              \
    } NAME

#define QUAD_LIGHT_RECORD(NAME)                                 \
    layout(shaderRecordEXT, scalar) buffer QuadLightRecord {    \
        vec3 emission;                                          \
        float unit_scale;                                       \
        float area_pdf;                                         \
        vec3 normal_ws;                                         \
        vec3 corner_ws;                                         \
        vec3 edge0_ws;                                          \
        vec3 edge1_ws;                                          \
    } NAME

#define SPHERE_LIGHT_RECORD(NAME)                               \
    layout(shaderRecordEXT, scalar) buffer SphereLightRecord {  \
        vec3 emission;                                          \
        float unit_scale;                                       \
        vec3 centre_ws;                                         \
        float radius_ws;                                        \
    } NAME

#define DOME_LIGHT_RECORD(NAME)                                 \
    layout(shaderRecordEXT, scalar) buffer DomeLightRecord {    \
        vec3 emission;                                          \
    } NAME

#define SOLID_ANGLE_LIGHT_RECORD(NAME)                              \
    layout(shaderRecordEXT, scalar) buffer SolidAngleLightRecord {  \
        vec3 emission;                                              \
        vec3 direction_ws;                                          \
        float solid_angle;                                          \
    } NAME

#define SPHERE_LIGHT_SIN_THETA_MIN          0.01f

#define QUAD_LIGHT_IS_TWO_SIDED             true

#define CALLABLE_SHADER_COUNT_PER_LIGHT     2

struct LightEvalParams {
    vec3 a;     //  in: position_or_extdir      out: emission
    vec3 b;     //  in: target_position         out: (solid_angle_pdf, -, -)
};

vec3 get_position_or_extdir(LightEvalParams p)  { return p.a; }
vec3 get_target_position(LightEvalParams p)     { return p.b; }

LightEvalParams write_light_eval_outputs(
    vec3 emission,
    float solid_angle_pdf)
{
    LightEvalParams p;
    p.a = emission;
    p.b.x = solid_angle_pdf;
    return p;
}

#define LIGHT_EVAL_SHADER_INDEX(LIGHT)      ((LIGHT)*CALLABLE_SHADER_COUNT_PER_LIGHT + 0)
#define LIGHT_EVAL_CALLABLE_INDEX           0
#define LIGHT_EVAL_DATA(NAME)               layout(location = LIGHT_EVAL_CALLABLE_INDEX) callableDataEXT LightEvalParams NAME
#define LIGHT_EVAL_DATA_IN(NAME)            layout(location = LIGHT_EVAL_CALLABLE_INDEX) callableDataInEXT LightEvalParams NAME

struct LightSampleParams {
    vec3 a;     // in: target position          out: position_or_extdir
    vec3 b;     // in: target normal            out: emission
    vec3 c;     // in: random numbers           out: (normal, solid_angle_pdf (ext in sign), unit_scale)
};

vec3 get_target_position(LightSampleParams p)   { return p.a; }
vec3 get_target_normal(LightSampleParams p)     { return p.b; }
vec2 get_rand_u01(LightSampleParams p)          { return p.c.xy; }

LightSampleParams write_light_sample_outputs(
    vec3 position_or_extdir,
    vec3 normal,
    vec3 emission,
    float solid_angle_pdf,
    bool is_external,
    float unit_scale)
{
    LightSampleParams p;
    p.a = position_or_extdir;
    p.b = emission;
    p.c = vec3(
        uintBitsToFloat(oct32_from_vec(normal)),
        is_external ? (-solid_angle_pdf) : solid_angle_pdf,
        unit_scale);
    return p;
}    

#define LIGHT_SAMPLE_SHADER_INDEX(LIGHT)    ((LIGHT)*CALLABLE_SHADER_COUNT_PER_LIGHT + 1)
#define LIGHT_SAMPLE_CALLABLE_INDEX         1
#define LIGHT_SAMPLE_DATA(NAME)             layout(location = LIGHT_SAMPLE_CALLABLE_INDEX) callableDataEXT LightSampleParams NAME
#define LIGHT_SAMPLE_DATA_IN(NAME)          layout(location = LIGHT_SAMPLE_CALLABLE_INDEX) callableDataInEXT LightSampleParams NAME
