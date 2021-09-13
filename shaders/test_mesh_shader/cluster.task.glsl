#version 460 core
#extension GL_EXT_scalar_block_layout : require
#extension GL_NV_mesh_shader : require
#extension GL_KHR_shader_subgroup_ballot: require

#extension GL_GOOGLE_include_directive : require
#include "cluster_common.glsl"

layout(local_size_x_id = TASK_GROUP_SIZE_ID) in;

CLUSTER_TASK(out, o_task);

vec4 transform_sphere(mat4 new_from_old, float new_from_old_scale, vec4 sphere)
{
    return vec4(
        (new_from_old * vec4(sphere.xyz, 1.0f)).xyz,
        new_from_old_scale * sphere.w
    );
}

vec4 transform_cone(mat3 new_from_old_rotflip, vec4 cone)
{
    return vec4(
        new_from_old_rotflip * cone.xyz,
        cone.w
    );
}

vec4 flip_cone(vec4 cone)
{
    return vec4(-cone.xyz, cone.w);
}

vec4 cone_from_sphere(vec4 sphere)
{
    vec3 sphere_centre = sphere.xyz;
    float sphere_radius = abs(sphere.w);
    float centre_dist = length(sphere_centre);
    float sin_theta = sphere_radius / centre_dist;
    if (sin_theta < 0.999f) {
        return vec4(
            normalize(sphere_centre),
            sqrt(max(0.0f, 1.0f - sin_theta * sin_theta))
        );
    } else {
        return vec4(1.0f, 0.0f, 0.0f, -1.0f);
    }
}

vec4 expand_cone_by_90_degrees(vec4 cone)
{
    float cos_a = cone.w;
    if (cos_a > 0.0f) {
        float sin_a = sqrt(max(0.0f, 1.0f - cos_a*cos_a));
        cone.w = -sin_a;
    } else {
        cone.w = -1.0f;
    }
    return cone;
}

bool cones_intersect(vec4 a, vec4 b)
{
    float cos_c = dot(a.xyz, b.xyz);

    float cos_a = a.w;
    float sin_a = sqrt(max(0.0f, 1.0f - cos_a*cos_a));

    float cos_b = b.w;
    float sin_b = sqrt(max(0.0f, 1.0f - cos_b*cos_b));

    float cos_ab = cos_a*cos_b - sin_a*sin_b;
    float sin_ab = sin_a*cos_b + cos_a*sin_b;

    return (cos_a < cos_c) || (cos_b < cos_c) || (sin_ab < 0.0f) || (cos_ab < cos_c);
}

void main()
{
    uint task_index = gl_GlobalInvocationID.x;
    uint task_index_within_group = gl_LocalInvocationID.x;

    bool is_valid = (task_index < g_cluster.task_count);
    if (is_valid) {
        vec4 position_sphere = transform_sphere(
            g_cluster.view_from_local,
            g_cluster.view_from_local_scale,
            g_cluster_desc.arr[task_index].position_sphere);
        vec4 position_cone = cone_from_sphere(position_sphere);
        vec4 view_cone = flip_cone(position_cone);

        vec4 face_normal_cone = transform_cone(
            g_cluster.view_from_local_rotflip,
            g_cluster_desc.arr[task_index].face_normal_cone);
        vec4 visibility_cone = expand_cone_by_90_degrees(face_normal_cone);

        if (!cones_intersect(view_cone, visibility_cone)) {
            is_valid = false;
        }
    }

    uvec4 valid_mask = subgroupBallot(is_valid);
    uint valid_count = subgroupBallotBitCount(valid_mask);

    uint valid_index = subgroupBallotExclusiveBitCount(valid_mask);
    if (is_valid) {
        o_task.task_index[valid_index] = task_index;
    }
    gl_TaskCountNV = valid_count;
}
