#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "extend_common.glsl"

EXTEND_PAYLOAD_IN(g_extend);

void main()
{
    g_extend.info = create_miss_info();
    g_extend.position_or_extdir = gl_WorldRayDirectionEXT;
}
