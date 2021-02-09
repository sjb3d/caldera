#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "extend_common.glsl"

layout(shaderRecordEXT, scalar) buffer ExtendMissRecord {
    ExtendShader shader;
} g_record;

EXTEND_PAYLOAD_IN(g_extend);

void main()
{
    const bool is_emissive = ((g_record.shader.flags & EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT) != 0);

    g_extend.position = gl_WorldRayDirectionEXT;
    g_extend.normal_oct32 = 0;
    g_extend.hit = create_miss_data(is_emissive, g_record.shader.light_index);
}
