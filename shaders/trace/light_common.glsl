#define LIGHT_UNIFORM_DATA(NAME)                                    \
    layout(set = 0, binding = 1, scalar) uniform LightUniforms {    \
        uint sample_sphere_solid_angle;                             \
        uint sampled_light_count;                                   \
    } NAME

#define QUAD_LIGHT_RECORD(NAME)                                 \
    layout(shaderRecordEXT, scalar) buffer QuadLightRecord {    \
        vec3 emission;                                          \
        float unit_value;                                       \
        float area_pdf;                                         \
        vec3 normal_ws;                                         \
        vec3 corner_ws;                                         \
        vec3 edge0_ws;                                          \
        vec3 edge1_ws;                                          \
    } NAME

#define SPHERE_LIGHT_RECORD(NAME)                               \
    layout(shaderRecordEXT, scalar) buffer SphereLightRecord {  \
        vec3 emission;                                          \
        float unit_value;                                       \
        vec3 centre_ws;                                         \
        float radius_ws;                                        \
    } NAME

#define CALLABLE_SHADER_COUNT_PER_LIGHT       2

struct LightEvalData {
    vec3 position;          // input only
    vec3 normal;            // input: target normal
    vec3 emission;          // input: target position
    float solid_angle_pdf;
};

#define LIGHT_EVAL_SHADER_INDEX(LIGHT)      (CALLABLE_SHADER_COUNT_PER_LIGHT*(LIGHT) + 0)
#define LIGHT_EVAL_CALLABLE_INDEX           0
#define LIGHT_EVAL_DATA(NAME)               layout(location = 0) callableDataEXT LightEvalData NAME
#define LIGHT_EVAL_DATA_IN(NAME)            layout(location = 0) callableDataInEXT LightEvalData NAME

struct LightSampleData {
    vec3 position;          // input: target position
    vec3 normal;            // input: target normal
    vec3 emission;          // input: random numbers
    float solid_angle_pdf;
    float unit_value;
};

#define LIGHT_SAMPLE_SHADER_INDEX(LIGHT)    (CALLABLE_SHADER_COUNT_PER_LIGHT*(LIGHT) + 1)
#define LIGHT_SAMPLE_CALLABLE_INDEX         1
#define LIGHT_SAMPLE_DATA(NAME)             layout(location = 1) callableDataEXT LightSampleData NAME
#define LIGHT_SAMPLE_DATA_IN(NAME)          layout(location = 1) callableDataInEXT LightSampleData NAME
