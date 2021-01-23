#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec3 a_pos;

layout(scalar, set = 0, binding = 0) uniform TestData {
    mat4 proj_from_local;
} g_test;

out gl_PerVertex {
    vec4 gl_Position;
};

void main()
{
    gl_Position = g_test.proj_from_local * vec4(a_pos, 1.f);
}
