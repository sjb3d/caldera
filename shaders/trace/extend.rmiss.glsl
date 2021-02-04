#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "record.glsl"
#include "payload.glsl"

layout(shaderRecordEXT, scalar) buffer ExtendMissRecord {
    uint flags;
} g_record;

EXTEND_PAYLOAD_WRITE(g_extend);

void main()
{
    const bool is_emissive = ((g_record.flags & EXTEND_RECORD_FLAGS_IS_EMISSIVE_BIT) != 0);
    g_extend.hit = create_miss_data(is_emissive);
}
