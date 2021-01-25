#version 460 core

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D g_ids;

void main()
{
    const uint id = imageLoad(g_ids, ivec2(gl_FragCoord.xy)).x;
    const vec3 col = (id == 0) ? vec3(0.f) : unpackUnorm4x8(hash(id)).xyz;

    o_col = vec4(col, 1.f);
}
