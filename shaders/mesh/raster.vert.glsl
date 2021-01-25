#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec3 a_pos;
layout(location = 1) in mat3x4 a_world_from_local_transpose;

layout(scalar, set = 0, binding = 0) uniform TestData {
    mat4 proj_from_world;
} g_test;

out gl_PerVertex {
    vec4 gl_Position;
};

void main()
{
    const vec3 pos_world_space = vec4(a_pos, 1.f) * a_world_from_local_transpose;

    gl_Position = g_test.proj_from_world * vec4(pos_world_space, 1.f);
}
