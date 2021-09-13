const uint TASK_GROUP_SIZE = 32;

const uint MAX_VERTICES_PER_CLUSTER = 64;
const uint MAX_TRIANGLES_PER_CLUSTER = 124;
const uint MAX_PACKED_INDICES_PER_CLUSTER = (MAX_TRIANGLES_PER_CLUSTER * 3) / 4;

struct ClusterDesc
{
    // TODO: bounds
    uint vertex_count;
    uint triangle_count;
    uint vertices[MAX_VERTICES_PER_CLUSTER];
    uint packed_indices[MAX_PACKED_INDICES_PER_CLUSTER];
};

#define CLUSTER_TASK(DIR, NAME) taskNV DIR Task {   \
    uint id[TASK_GROUP_SIZE];                       \
} NAME;

layout(set = 0, binding = 0, scalar) uniform ClusterUniforms {
    uint task_count;
} g_cluster;

layout(set = 0, binding = 1, scalar) readonly buffer ClusterDescArr {
    ClusterDesc arr[];
};
