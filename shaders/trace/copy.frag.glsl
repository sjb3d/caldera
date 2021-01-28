#version 460 core

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D g_trace_output;

void main()
{
    const uint trace_output = imageLoad(g_trace_output, ivec2(gl_FragCoord.xy)).x;
    const vec3 col = (trace_output == 0) ? vec3(.1f) : unpackUnorm4x8(hash(trace_output)).xyz;

    o_col = vec4(col, 1.f);
}
