#ifndef INCLUDED_LIGHT_COMMON
#define INCLUDED_LIGHT_COMMON

#include "normal_pack.glsl"

struct LightAliasEntry {
    float split;
    uint indices;
};
layout(buffer_reference, scalar) buffer LightAliasTable {
    LightAliasEntry entries[];
};

struct LightInfoEntry {
    uint light_type;
    float probability;
    uint params_offset;
};
layout(buffer_reference, scalar) buffer LightInfoTable {
    LightInfoEntry entries[];
};

#define LIGHT_TYPE_QUAD         0
#define LIGHT_TYPE_DISC         1
#define LIGHT_TYPE_SPHERE       2
#define LIGHT_TYPE_DOME         3
#define LIGHT_TYPE_SOLID_ANGLE  4

struct PlanarLightParams {
    vec3 emission;
    float unit_scale;
    float area_pdf;
    vec3 normal_ws;
    vec3 point_ws;
    vec3 vec0_ws;
    vec3 vec1_ws;
};
layout(buffer_reference, scalar) buffer PlanarLightParamsBuffer {
    PlanarLightParams params;
};

struct SphereLightParams {
    vec3 emission;
    float unit_scale;
    vec3 centre_ws;
    float radius_ws;
};
layout(buffer_reference, scalar) buffer SphereLightParamsBuffer {
    SphereLightParams params;
};

struct DomeLightParams {
    vec3 emission;
};
layout(buffer_reference, scalar) buffer DomeLightParamsBuffer {
    DomeLightParams params;
};

struct SolidAngleLightParams {
    vec3 emission;
    vec3 direction_ws;
    float solid_angle;
};
layout(buffer_reference, scalar) buffer SolidAngleLightParamsBuffer {
    SolidAngleLightParams params;
};

#define SPHERE_LIGHT_SIN_THETA_MIN          0.01f

#endif
