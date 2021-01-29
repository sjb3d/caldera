#version 460 core
#extension GL_EXT_ray_tracing : require

#extension GL_GOOGLE_include_directive : require
#include "payload.glsl"

EXTEND_PAYLOAD_WRITE(g_payload);

void main()
{
    g_payload.packed_normal = 0;
    g_payload.packed_shader = 0;
}
