
const uint MAX_VERTICES_PER_CLUSTER = 64;
const uint MAX_TRIANGLES_PER_CLUSTER = 124;
const uint MAX_PACKED_INDICES_PER_CLUSTER = (MAX_TRIANGLES_PER_CLUSTER * 3) / 4;

struct ClusterDesc
{
    vec4 position_sphere;
    vec4 face_normal_cone;
    uint vertex_count;
    uint triangle_count;
    uint vertices[MAX_VERTICES_PER_CLUSTER];
    uint packed_indices[MAX_PACKED_INDICES_PER_CLUSTER];
};

#define TASK_GROUP_SIZE_ID 0
layout(constant_id = TASK_GROUP_SIZE_ID) const int task_group_size = 1;

#define CLUSTER_TASK(DIR, NAME) taskNV DIR Task {   \
    uint task_index[task_group_size];               \
} NAME;

layout(set = 0, binding = 0, scalar) uniform ClusterUniforms {
    mat4 proj_from_local;
    mat4 view_from_local;
    mat3 view_from_local_rotflip;
    float view_from_local_scale;
    uint task_count;
} g_cluster;

layout(set = 0, binding = 1, scalar) readonly buffer PositionArr {
    vec3 arr[3];
} g_position;

layout(set = 0, binding = 2, scalar) readonly buffer NormalArr {
    vec3 arr[3];
} g_normal;

layout(set = 0, binding = 3, scalar) readonly buffer ClusterDescArr {
    ClusterDesc arr[];
} g_cluster_desc;
