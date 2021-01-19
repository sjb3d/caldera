#version 430 core

#extension GL_GOOGLE_include_directive : require
#include "sampler.glsl"

layout(location = 0) out vec4 o_col;

void main()
{
    const uint id = gl_PrimitiveID;
    const vec3 col = unpackUnorm4x8(hash(id)).xyz;

    o_col = vec4(col, 0.f);
}
