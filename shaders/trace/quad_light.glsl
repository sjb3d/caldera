#include "light_common.glsl"

void planar_light_eval(
    PlanarLightParams params,
    bool is_two_sided,
    vec3 target_position,
    vec3 light_position,
    out vec3 emission,
    out float solid_angle_pdf)
{
    const vec3 light_normal = params.normal_ws;
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    emission = (is_two_sided || facing_term > 0.f) ? params.emission : vec3(0.f);

    const float distance_sq = dot(target_from_light, target_from_light);
    solid_angle_pdf = solid_angle_pdf_from_area_pdf(params.area_pdf, facing_term, distance_sq);
}

void quad_light_sample(
    PlanarLightParams params,
    bool is_two_sided,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_position,
    out Normal32 light_normal_packed,
    out vec3 emission,
    out float solid_angle_pdf_and_ext_bit,
    out float unit_scale)
{
    light_position = params.point_ws + light_rand_u01.x*params.vec0_ws + light_rand_u01.y*params.vec1_ws;
    
    const vec3 light_normal = params.normal_ws;
    light_normal_packed = make_normal32(light_normal);

    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    emission = (is_two_sided || facing_term > 0.f) ? params.emission : vec3(0.f);

    const float distance_sq = dot(target_from_light, target_from_light);
    solid_angle_pdf_and_ext_bit = solid_angle_pdf_from_area_pdf(params.area_pdf, facing_term, distance_sq);

    unit_scale = params.unit_scale;
}
