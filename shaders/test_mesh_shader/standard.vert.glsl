#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;

layout(set = 0, binding = 0, scalar) uniform StandardUniforms {
    mat4 proj_from_local;
} g_standard;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) out vec3 v_color;

void main()
{
    gl_Position = g_standard.proj_from_local * vec4(a_pos, 1.f);
    v_color = 0.5f*a_normal + 0.5f;
}
