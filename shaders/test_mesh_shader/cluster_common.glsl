const uint TASK_GROUP_SIZE = 32;

#define CLUSTER_TASK(DIR, NAME) taskNV DIR Task {   \
    uint id[TASK_GROUP_SIZE];                   \
} NAME;

layout(set = 0, binding = 0, scalar) uniform ClusterUniforms {
    uint task_count;
} g_cluster;
