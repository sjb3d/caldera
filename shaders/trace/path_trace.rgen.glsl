#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "payload.glsl"
#include "normal_pack.glsl"

layout(scalar, set = 0, binding = 0) uniform PathTraceData {
    vec3 ray_origin;
    mat3x3 ray_vec_from_coord;
    mat4x3 world_from_light;
    vec2 light_size_ws;
} g_data;
layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage2D g_output;

EXTEND_PAYLOAD_READ(g_extend);

void main()
{
    const vec3 ray_vec = g_data.ray_vec_from_coord * vec3(gl_LaunchIDEXT.xy + .5f, 1.f);
    const vec3 ray_dir = normalize(ray_vec);

    traceRayEXT(
        g_accel,
        gl_RayFlagsOpaqueEXT,
        0xff,
        0,
        1,
        0,
        g_data.ray_origin,
        0.0,
        ray_dir,
        1000.f,
        0);

    uint output_value = 0;
    if (g_extend.is_valid != 0) {
        const vec3 to_light_dir = normalize(g_data.world_from_light[3] - g_extend.position);
        const vec3 normal_ws = normalize(vec_from_oct32(g_extend.normal_oct32));
        const float facing_term = .5f*dot(normal_ws, to_light_dir) + .5f;

        const vec3 reflectance = vec3(
            unpackHalf2x16(g_extend.reflectance_and_emission.x),
            unpackHalf2x16(g_extend.reflectance_and_emission.y).x);

        output_value = packUnorm4x8(vec4(reflectance, facing_term));
    }

    imageStore(g_output, ivec2(gl_LaunchIDEXT.xy), uvec4(output_value, 0, 0, 0));
}
