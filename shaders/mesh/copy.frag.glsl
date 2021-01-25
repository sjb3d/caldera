#version 460 core

#extension GL_GOOGLE_include_directive : require
#include "normal_pack.glsl"

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D g_trace_output;

void main()
{
    const uint trace_output = imageLoad(g_trace_output, ivec2(gl_FragCoord.xy)).x;
    const vec3 normal = normalize(vec_from_oct32(trace_output));

    const vec3 col = (trace_output == 0) ? vec3(0.f) : (.5f*normal + .5f);

    o_col = vec4(col, 1.f);
}
