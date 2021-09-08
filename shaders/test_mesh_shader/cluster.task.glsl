#version 460 core
#extension GL_EXT_scalar_block_layout : require
#extension GL_NV_mesh_shader : require

layout(local_size_x = 3) in;

taskNV out Task {
    uint id[3];
} o_task;

void main()
{
    uint thread_idx = gl_LocalInvocationID.x;

    o_task.id[thread_idx] = thread_idx;

    gl_TaskCountNV = 3;
}
