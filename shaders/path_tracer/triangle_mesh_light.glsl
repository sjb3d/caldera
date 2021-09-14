#include "light_common.glsl"

void triangle_mesh_light_eval(
    uint64_t params_addr,
    uint triangle_index,
    vec3 target_position,
    vec3 light_position,
    out vec3 illuminant_tint,
    out float solid_angle_pdf)
{
    TriangleMeshLightParamsBuffer buf = TriangleMeshLightParamsBuffer(params_addr);

    const uvec3 tri = buf.params.index_buffer.tri[triangle_index];
    const vec3 p0 = buf.params.position_buffer.pos[tri.x];
    const vec3 p1 = buf.params.position_buffer.pos[tri.y];
    const vec3 p2 = buf.params.position_buffer.pos[tri.z];

    const vec3 light_normal = normalize(cross(p2 - p1, p0 - p1));
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    illuminant_tint = (facing_term > 0.f) ? buf.params.illuminant_tint : vec3(0.f);

    const float distance_sq = dot(target_from_light, target_from_light);
    solid_angle_pdf = solid_angle_pdf_from_area_pdf(buf.params.area_pdf, facing_term, distance_sq);
}

void triangle_mesh_light_sample(
    uint64_t params_addr,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_position,
    out Normal32 light_normal_packed,
    out vec3 illuminant_tint,
    out float solid_angle_pdf_and_ext_bit,
    out float unit_scale)
{
    TriangleMeshLightParamsBuffer buf = TriangleMeshLightParamsBuffer(params_addr);

    const uint entry_index = sample_uniform_discrete(buf.params.triangle_count, light_rand_u01.x);
    const LightAliasEntry entry = buf.params.alias_table.entries[entry_index];
    uint triangle_index;
    if (split_random_variable(entry.split, light_rand_u01.y)) {
        triangle_index = entry.indices & 0xffffU;
    } else {
        triangle_index = entry.indices >> 16;
    }

    const uvec3 tri = buf.params.index_buffer.tri[triangle_index];
    const vec3 p0 = buf.params.position_buffer.pos[tri.x];
    const vec3 p1 = buf.params.position_buffer.pos[tri.y];
    const vec3 p2 = buf.params.position_buffer.pos[tri.z];

    vec2 bary_coord = light_rand_u01;
    if (bary_coord.x + bary_coord.y > 1.f) {
        bary_coord = 1.f - bary_coord;
    }
    light_position = p0*bary_coord.x + p1*bary_coord.y + p2*(1.f - bary_coord.x - bary_coord.y);
    
    const vec3 light_normal = normalize(cross(p2 - p1, p0 - p1));
    light_normal_packed = make_normal32(light_normal);

    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    illuminant_tint = (facing_term > 0.f) ? buf.params.illuminant_tint : vec3(0.f);

    const float distance_sq = dot(target_from_light, target_from_light);
    solid_angle_pdf_and_ext_bit = solid_angle_pdf_from_area_pdf(buf.params.area_pdf, facing_term, distance_sq);

    unit_scale = buf.params.unit_scale;
}
