#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "payload.glsl"
#include "normal_pack.glsl"

layout(set = 0, binding = 0, scalar) uniform PathTraceData {
    vec3 ray_origin;
    mat3x3 ray_vec_from_coord;
    mat4x3 world_from_light;
    vec2 light_size_ws;
} g_data;
layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage2D g_output;

EXTEND_PAYLOAD_READ(g_extend);
OCCLUSION_PAYLOAD_READ(g_occlusion);

void main()
{
    const vec3 ray_vec = g_data.ray_vec_from_coord * vec3(gl_LaunchIDEXT.xy + .5f, 1.f);
    const vec3 ray_dir = normalize(ray_vec);

    traceRayEXT(
        g_accel,
        gl_RayFlagsNoneEXT,
        0xff,
        EXTEND_HIT_SHADER_OFFSET,
        HIT_SHADER_COUNT_PER_INSTANCE,
        EXTEND_MISS_SHADER_OFFSET,
        g_data.ray_origin,
        0.f,
        ray_dir,
        1000.f,
        EXTEND_PAYLOAD_INDEX);

    uint output_value = 0;
    if ((g_extend.flags_packed & EXTEND_FLAGS_VALID_BIT) != 0) {
        const int log2_epsilon_factor = -18;
        const float hit_epsilon = ldexp(1.f, int(g_extend.flags_packed & EXTEND_FLAGS_MAX_EXP_MASK) - 128 + log2_epsilon_factor);

        const vec3 normal_ws = normalize(vec_from_oct32(g_extend.normal_oct32));
        const vec3 adjusted_hit_pos = g_extend.position + normal_ws*hit_epsilon;

        const float light_epsilon = ldexp(max_element(abs(g_data.world_from_light[3])) + max_element(abs(g_data.light_size_ws)), log2_epsilon_factor);
        const vec3 adjusted_light_pos = g_data.world_from_light[3] + g_data.world_from_light[2]*light_epsilon;

        const vec3 to_light_vec = adjusted_light_pos - adjusted_hit_pos;
        const float to_light_distance = length(to_light_vec);
        const vec3 to_light_dir = to_light_vec/to_light_distance;

        const float distance_epsilon_factor = 1.f - ldexp(1.f, log2_epsilon_factor);        
        traceRayEXT(
            g_accel,
            gl_RayFlagsNoneEXT | gl_RayFlagsTerminateOnFirstHitEXT,
            0xff,
            OCCLUSION_HIT_SHADER_OFFSET,
            HIT_SHADER_COUNT_PER_INSTANCE,
            OCCLUSION_MISS_SHADER_OFFSET,
            adjusted_hit_pos,
            0.f,
            to_light_dir,
            to_light_distance*distance_epsilon_factor,
            OCCLUSION_PAYLOAD_INDEX);

        const float facing_term = (g_occlusion.is_occluded != 0) ? 0.f : .5f*dot(normal_ws, to_light_dir) + .5f;

        const vec3 reflectance = vec3(
            unpackHalf2x16(g_extend.reflectance_and_emission.x),
            unpackHalf2x16(g_extend.reflectance_and_emission.y).x);

        output_value = packUnorm4x8(vec4(reflectance, facing_term));
    }

    imageStore(g_output, ivec2(gl_LaunchIDEXT.xy), uvec4(output_value, 0, 0, 0));
}
