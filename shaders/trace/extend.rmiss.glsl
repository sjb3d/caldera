#version 460 core
#extension GL_EXT_ray_tracing : require

#extension GL_GOOGLE_include_directive : require
#include "payload.glsl"

EXTEND_PAYLOAD_WRITE(g_extend);

void main()
{
    g_extend.is_valid = 0;
}
