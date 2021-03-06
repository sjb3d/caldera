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

struct LightFlags {
    uint bits;
};

uint get_light_type(LightFlags flags)           { return flags.bits & 0xffU; }
uint get_illuminant_index(LightFlags flags)     { return flags.bits >> 8; }

struct LightInfoEntry {
    LightFlags light_flags;
    float probability;
    uint params_offset;
};
layout(buffer_reference, scalar) buffer LightInfoTable {
    LightInfoEntry entries[];
};

#define LIGHT_TYPE_TRIANGLE_MESH    0
#define LIGHT_TYPE_QUAD             1
#define LIGHT_TYPE_DISC             2
#define LIGHT_TYPE_SPHERE           3
#define LIGHT_TYPE_DOME             4
#define LIGHT_TYPE_SOLID_ANGLE      5

layout(buffer_reference, scalar) buffer LightProbabilityTable {
    float probability[];
};
layout(buffer_reference, scalar) buffer LightIndexBuffer {
    uvec3 tri[];
};
layout(buffer_reference, scalar) buffer LightPositionBuffer {
    vec3 pos[];
};
struct TriangleMeshLightParams {
    LightAliasTable alias_table;
    LightIndexBuffer index_buffer;
    LightPositionBuffer position_buffer;
    mat4x3 world_from_local;
    vec3 illuminant_tint;
    uint triangle_count;
    float area_pdf;
    float unit_scale;
};
layout(buffer_reference, scalar) buffer TriangleMeshLightParamsBuffer {
    TriangleMeshLightParams params;
};

struct PlanarLightParams {
    vec3 illuminant_tint;
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
    vec3 illuminant_tint;
    float unit_scale;
    vec3 centre_ws;
    float radius_ws;
};
layout(buffer_reference, scalar) buffer SphereLightParamsBuffer {
    SphereLightParams params;
};

struct DomeLightParams {
    vec3 illuminant_tint;
};
layout(buffer_reference, scalar) buffer DomeLightParamsBuffer {
    DomeLightParams params;
};

struct SolidAngleLightParams {
    vec3 illuminant_tint;
    vec3 direction_ws;
    float solid_angle;
};
layout(buffer_reference, scalar) buffer SolidAngleLightParamsBuffer {
    SolidAngleLightParams params;
};

#define SPHERE_LIGHT_SIN_THETA_MIN          0.01f

#endif
