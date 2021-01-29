#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "normal_pack.glsl"

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 tri[];
};

layout(buffer_reference, scalar) buffer AttributeBuffer {
    vec3 normals[];
};

layout(shaderRecordEXT, scalar) buffer ShaderRecordData {
    IndexBuffer indices;
    AttributeBuffer attribs;
} g_record;

hitAttributeEXT vec2 g_bary_coord;

layout(location = 0) rayPayloadInEXT uint g_payload;

void main()
{
    const uvec3 tri = g_record.indices.tri[gl_PrimitiveID];
    const vec3 n0 = g_record.attribs.normals[tri[0]];
    const vec3 n1 = g_record.attribs.normals[tri[1]];
    const vec3 n2 = g_record.attribs.normals[tri[2]];

    const vec3 normal = normalize(
          n0*(1.f - g_bary_coord.x - g_bary_coord.y)
        + n1*g_bary_coord.x
        + n2*g_bary_coord.y);
    const uint normal_oct32 = oct32_from_vec(normal);

    // avoid zero since we want that to mean "miss"
    g_payload = (normal_oct32 == 0) ? 1 : normal_oct32;
}
