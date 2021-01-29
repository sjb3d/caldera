#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "payload.glsl"
#include "normal_pack.glsl"

layout(buffer_reference, scalar) buffer TriangleNormalBuffer {
    uint packed_normals[];
};

layout(shaderRecordEXT, scalar) buffer ShaderRecordData {
    TriangleNormalBuffer triangle_normal_buffer;
    uint packed_shader;
} g_record;

EXTEND_PAYLOAD_WRITE(g_extend);

void main()
{   
    // transform normal vector to world space
    const vec3 face_normal_vec_ls = vec_from_oct32(g_record.triangle_normal_buffer.packed_normals[gl_PrimitiveID]);
    const vec3 hit_normal_vec_ls
        = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT)
        ? face_normal_vec_ls
        : -face_normal_vec_ls;
    const vec3 hit_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_normal_vec_ls, 0.f);

    g_extend.packed_normal = oct32_from_vec(hit_normal_vec_ws);
    g_extend.packed_shader = g_record.packed_shader;
}
