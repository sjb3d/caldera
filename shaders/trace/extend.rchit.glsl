#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "payload.glsl"
#include "normal_pack.glsl"

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 tri[];
};

layout(buffer_reference, scalar) buffer PositionBuffer {
    vec3 pos[];
};

layout(shaderRecordEXT, scalar) buffer ShaderRecordData {
    IndexBuffer index_buffer;
    PositionBuffer position_buffer;
    uvec3 reflectance_and_emission;
} g_record;

hitAttributeEXT vec2 g_bary_coord;

EXTEND_PAYLOAD_WRITE(g_extend);

void main()
{   
    const uvec3 tri = g_record.index_buffer.tri[gl_PrimitiveID];
    const vec3 p0 = g_record.position_buffer.pos[tri[0]];
    const vec3 p1 = g_record.position_buffer.pos[tri[1]];
    const vec3 p2 = g_record.position_buffer.pos[tri[2]];

    const vec3 face_normal_vec_ls = cross(p1 - p0, p2 - p0);
    const vec3 hit_pos_ls
        = p0*(1.f - g_bary_coord.x - g_bary_coord.y)
        + p1*g_bary_coord.x
        + p2*g_bary_coord.y
        ;

    // transform normal vector to world space
    const vec3 hit_normal_vec_ls
        = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT)
        ? face_normal_vec_ls
        : -face_normal_vec_ls;
    const vec3 hit_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_normal_vec_ls, 0.f);
    const vec3 hit_pos_ws = gl_ObjectToWorldEXT * vec4(hit_pos_ls, 1.f);

    // estimate floating point number size for local and world space
    // TODO: handle scale
    const vec3 max_offset = max(max(abs(p0), abs(p1)), abs(p2)) + abs(gl_ObjectToWorldEXT[3]);
    int max_exponent = 0;
    frexp(max_element(max_offset), max_exponent);
    uint max_exponent_packed = uint(max_exponent + 128);

    g_extend.position = hit_pos_ws;
    g_extend.normal_oct32 = oct32_from_vec(hit_normal_vec_ws);
    g_extend.reflectance_and_emission = g_record.reflectance_and_emission;
    g_extend.flags_packed
        = (max_exponent_packed & EXTEND_FLAGS_MAX_EXP_MASK)
        | EXTEND_FLAGS_HAS_SURFACE_BIT
        ;
}