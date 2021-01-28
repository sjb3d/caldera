#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

layout(scalar, set = 0, binding = 0) uniform CameraData {
    vec3 ray_origin;
    mat3x3 ray_vec_from_coord;
} g_camera;
layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage2D g_output;

layout(location = 0) rayPayloadEXT uint g_payload;

void main()
{
    const vec3 ray_vec = g_camera.ray_vec_from_coord * vec3(gl_LaunchIDEXT.xy, 1.f);
    const vec3 ray_dir = normalize(ray_vec);

    traceRayEXT(
        g_accel,
        gl_RayFlagsOpaqueEXT,
        0xff,
        0,
        0,
        0,
        g_camera.ray_origin,
        0.0,
        ray_dir,
        1000.f,
        0);

    imageStore(g_output, ivec2(gl_LaunchIDEXT.xy), uvec4(g_payload, 0, 0, 0));
}
