#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in mat3x4 a_world_from_local_transpose;

layout(set = 0, binding = 0, scalar) uniform RasterData {
    mat4 proj_from_world;
} g_raster;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) out vec3 v_normal;

void main()
{
    const vec3 pos_world_space = vec4(a_pos, 1.f) * a_world_from_local_transpose;

    gl_Position = g_raster.proj_from_world * vec4(pos_world_space, 1.f);
    v_normal = a_normal;
}
