#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "normal_pack.glsl"

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 tri[];
};

layout(buffer_reference, scalar) buffer PositionBuffer {
    vec3 pos[];
};

layout(shaderRecordEXT, scalar) buffer ExtendTriangleHitRecord {
    IndexBuffer index_buffer;
    PositionBuffer position_buffer;
    ExtendShader shader;
} g_record;

hitAttributeEXT vec2 g_bary_coord;

EXTEND_PAYLOAD_IN(g_extend);

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

    const uint bsdf_type = g_record.shader.flags & EXTEND_SHADER_FLAGS_BSDF_TYPE_MASK;
    const bool is_emissive = ((g_record.shader.flags & EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT) != 0);

    // estimate floating point number size for local and world space
    // TODO: handle scale
    const float max_position_value = max_element(
        max(max(abs(p0), abs(p1)), abs(p2))
        + abs(gl_ObjectToWorldEXT[3])
    );
    int max_exponent = 0;
    frexp(max_position_value, max_exponent);

    g_extend.position = hit_pos_ws;
    g_extend.normal_oct32 = oct32_from_vec(hit_normal_vec_ws);
    g_extend.hit = create_hit_data(
        bsdf_type,
        g_record.shader.reflectance,
        is_emissive,
        g_record.shader.light_index,
        max_exponent);
}
