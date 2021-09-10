#version 460 core
#extension GL_EXT_scalar_block_layout : require
#extension GL_NV_mesh_shader : require

#extension GL_GOOGLE_include_directive : require
#include "cluster_common.glsl"

layout(local_size_x = 1) in;

CLUSTER_TASK(in, i_task);

layout(triangles, max_vertices = 3, max_primitives = 1) out;

layout(location = 0) out vec3 v_color[];

void main()
{
    uint id = i_task.id[gl_WorkGroupID.x];
    vec2 offset;
    switch (id) {
        case 0: offset = vec2( 0.5,  0.5); break;
        case 1: offset = vec2( 0.0, -0.5); break;
        default:
        case 2: offset = vec2(-0.5,  0.5); break;
    }

    gl_MeshVerticesNV[0].gl_Position = vec4(offset + vec2( 0.5,  0.5), 0.0, 1.0);
    gl_MeshVerticesNV[1].gl_Position = vec4(offset + vec2( 0.0, -0.5), 0.0, 1.0);
    gl_MeshVerticesNV[2].gl_Position = vec4(offset + vec2(-0.5,  0.5), 0.0, 1.0);

    v_color[0] = vec3(1.0, 0.0, 0.0);
    v_color[1] = vec3(0.0, 1.0, 0.0);
    v_color[2] = vec3(0.0, 0.0, 1.0);

    gl_PrimitiveIndicesNV[0] = 0;
    gl_PrimitiveIndicesNV[1] = 1;
    gl_PrimitiveIndicesNV[2] = 2;

    gl_PrimitiveCountNV = 1;
}
