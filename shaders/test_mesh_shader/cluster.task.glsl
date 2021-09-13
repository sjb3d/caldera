#version 460 core
#extension GL_EXT_scalar_block_layout : require
#extension GL_NV_mesh_shader : require
#extension GL_KHR_shader_subgroup_ballot: require

#extension GL_GOOGLE_include_directive : require
#include "cluster_common.glsl"

layout(local_size_x_id = TASK_GROUP_SIZE_ID) in;

CLUSTER_TASK(out, o_task);

void main()
{
    uint task_index = gl_GlobalInvocationID.x;
    uint task_index_within_group = gl_LocalInvocationID.x;

    bool is_valid = (task_index < g_cluster.task_count);
    uvec4 valid_mask = subgroupBallot(is_valid);
    uint valid_count = subgroupBallotBitCount(valid_mask);
    
    if (is_valid) {
        o_task.id[task_index_within_group] = task_index;
    }
    gl_TaskCountNV = valid_count;
}
