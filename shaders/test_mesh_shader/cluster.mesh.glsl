#version 460 core
#extension GL_EXT_scalar_block_layout : require
#extension GL_NV_mesh_shader : require

#extension GL_GOOGLE_include_directive : require
#include "cluster_common.glsl"
#include "sampler.glsl"

layout(local_size_x = 32) in;

CLUSTER_TASK(in, i_task);

layout(triangles, max_vertices = MAX_VERTICES_PER_CLUSTER, max_primitives = MAX_TRIANGLES_PER_CLUSTER) out;

layout(location = 0) out vec3 v_color[];

void main()
{
    uint task_index = i_task.task_index[gl_WorkGroupID.x];
    uint vertex_count = g_cluster_desc.arr[task_index].vertex_count;
    uint triangle_count = g_cluster_desc.arr[task_index].triangle_count;

    for (uint index = gl_LocalInvocationID.x; index < vertex_count; index += gl_WorkGroupSize.x) {
        uint vertex = g_cluster_desc.arr[task_index].vertices[index];
        vec3 pos = g_position.arr[vertex];
        vec3 normal = g_normal.arr[vertex];
        gl_MeshVerticesNV[index].gl_Position = g_cluster.proj_from_local * vec4(pos, 1.0f);
        vec3 debug_color = unpackUnorm4x8(hash(task_index)).xyz;
        v_color[index] = debug_color + 0.01f*(0.5f*normal + 0.5f);
    }

    uint packed_index_count = ((triangle_count * 3) + 3) / 4;
    for (uint index = gl_LocalInvocationID.x; index < packed_index_count; index += gl_WorkGroupSize.x) {
        uint packed_indices = g_cluster_desc.arr[task_index].packed_indices[index];
        writePackedPrimitiveIndices4x8NV(4*index, packed_indices);
    }
    gl_PrimitiveCountNV = triangle_count;
}
