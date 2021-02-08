#version 460 core
#extension GL_EXT_ray_tracing : require

#extension GL_GOOGLE_include_directive : require
#include "occlusion_common.glsl"

OCCLUSION_PAYLOAD_IN(g_occlusion);

void main()
{
    g_occlusion.is_occluded = 0;
}
