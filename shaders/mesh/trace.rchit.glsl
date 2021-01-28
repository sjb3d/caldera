#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_GOOGLE_include_directive : require
#include "normal_pack.glsl"

layout(buffer_reference, buffer_reference_align=8, scalar) buffer IndexBuffer {
    uvec3 indices[];
};

layout(buffer_reference, buffer_reference_align=8, scalar) buffer AttributeBuffer {
    vec3 normals[];
};

layout(shaderRecordEXT, scalar) buffer ShaderRecordData {
    IndexBuffer faces;
    AttributeBuffer attribs;
} g_record;

hitAttributeEXT vec2 g_bary_coord;

layout(location = 0) rayPayloadInEXT uint g_payload;

void main()
{
    const uvec3 indices = g_record.faces.indices[gl_PrimitiveID];
    const vec3 n0 = g_record.attribs.normals[indices[0]];
    const vec3 n1 = g_record.attribs.normals[indices[1]];
    const vec3 n2 = g_record.attribs.normals[indices[2]];

    const vec3 normal = normalize(
          n0*(1.f - g_bary_coord.x - g_bary_coord.y)
        + n1*g_bary_coord.x
        + n2*g_bary_coord.y);
    const uint packed_normal = oct32_from_vec(normal);

    // avoid zero since we want that to mean "miss"
    g_payload = (packed_normal == 0) ? 1 : packed_normal;
}
