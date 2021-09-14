#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "transform.glsl"

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;

layout(set = 0, binding = 0, scalar) uniform StandardUniforms {
    mat4 proj_from_view;
    PackedTransform view_from_local;
} g_standard;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal_viewspace;
layout(location = 2) out vec3 v_pos_viewspace;

void main()
{
    vec3 pos_viewspace = transform_point(g_standard.view_from_local, a_pos);
    vec3 normal_viewspace = transform_unit(g_standard.view_from_local, a_normal);

    gl_Position = g_standard.proj_from_view * vec4(pos_viewspace, 1.f);
    v_color = vec3(.9f);
    v_normal_viewspace = normal_viewspace;
    v_pos_viewspace = pos_viewspace;
}
