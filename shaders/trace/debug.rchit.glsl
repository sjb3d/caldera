#version 460 core
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT uint g_payload;

void main()
{
    g_payload = 1 + (gl_InstanceCustomIndexEXT << 16) + (gl_GeometryIndexEXT << 8) + gl_PrimitiveID;
}
