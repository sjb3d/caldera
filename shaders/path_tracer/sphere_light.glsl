#include "light_common.glsl"

void sphere_light_eval(
    uint64_t params_addr,
    bool sample_solid_angle,
    vec3 target_position,
    vec3 light_position,
    out vec3 illuminant_tint,
    out float solid_angle_pdf)
{
    SphereLightParamsBuffer buf = SphereLightParamsBuffer(params_addr);

    const vec3 light_normal = normalize(light_position - buf.params.centre_ws);
    const vec3 target_from_light = target_position - light_position;
    const vec3 connection_dir = normalize(target_from_light);
    const float facing_term = dot(connection_dir, light_normal);
    illuminant_tint = (facing_term > 0.f) ? buf.params.illuminant_tint : vec3(0.f);

    if (sample_solid_angle) {
        const vec3 centre_from_target = buf.params.centre_ws - target_position;
        const float sin_theta = buf.params.radius_ws/length(centre_from_target);

        float solid_angle;
        if (sin_theta > SPHERE_LIGHT_SIN_THETA_MIN) {
            // accurate solid angle on a sphere
            const float cos_theta = sqrt(max(0.f, 1.f - sin_theta*sin_theta));
            solid_angle = 2.f*PI*(1.f - cos_theta);
        } else {
            // approximate with a disc
            solid_angle = PI*sin_theta*sin_theta;
        }

        solid_angle_pdf = 1.f/solid_angle;
    } else {
        const float distance_sq = dot(target_from_light, target_from_light);
        const float area_ws = 4.f * PI * buf.params.radius_ws * buf.params.radius_ws;
        solid_angle_pdf = solid_angle_pdf_from_area_pdf(1.f/area_ws, facing_term, distance_sq);
    }
}

void sphere_light_sample(
    uint64_t params_addr,
    bool sample_solid_angle,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_position,
    out Normal32 light_normal_packed,
    out vec3 illuminant_tint,
    out float solid_angle_pdf_and_ext_bit,
    out float unit_scale)
{
    SphereLightParamsBuffer buf = SphereLightParamsBuffer(params_addr);

    if (sample_solid_angle) {
        const vec3 centre_from_target = buf.params.centre_ws - target_position;
        const mat3 basis = basis_from_z_axis(normalize(centre_from_target));

        const float sin_theta = buf.params.radius_ws/length(centre_from_target);

        float solid_angle;
        vec3 sample_dir;
        if (sin_theta > SPHERE_LIGHT_SIN_THETA_MIN) {
            // compute solid angle on a sphere
            const float cos_theta = sqrt(max(0.f, 1.f - sin_theta*sin_theta));
            solid_angle = 2.f*PI*(1.f - cos_theta);

            // sample the visible solid angle
            const vec3 ray_dir_ls = sample_solid_angle_uniform(cos_theta, light_rand_u01);
            const vec3 ray_dir = basis*ray_dir_ls;

            // intersect ray with sphere for sample point
            const vec3 p = -centre_from_target;
            const vec3 d = ray_dir;
            const float r = buf.params.radius_ws;
            const vec2 t = ray_vs_sphere_force_hit(p, d, r);

            const float t_min = max(min_element(t), 0.f);

            sample_dir = normalize(p + t_min*d);
        } else {
            // approximate with a disc since it is more numerically stable
            solid_angle = PI*sin_theta*sin_theta;

            const vec2 disc_pos = sample_disc_uniform(light_rand_u01);
            const vec3 sample_vec_ls = vec3(disc_pos*sin_theta, -1.f);

            sample_dir = normalize(basis*sample_vec_ls);
        }

        // construct the sample here
        light_position = buf.params.centre_ws + buf.params.radius_ws*sample_dir;
        const vec3 light_normal = sample_dir;
        light_normal_packed = make_normal32(light_normal);
        const vec3 target_from_light = target_position - light_position;
        const float facing_term = dot(target_from_light, light_normal);
        illuminant_tint = (facing_term > 0.f) ? buf.params.illuminant_tint : vec3(0.f);

        solid_angle_pdf_and_ext_bit = 1.f/solid_angle;
        unit_scale = buf.params.unit_scale;
    } else {
        const vec3 sample_dir = sample_sphere_uniform(light_rand_u01);
        light_position = buf.params.centre_ws + buf.params.radius_ws*sample_dir;
        const vec3 light_normal = sample_dir;
        light_normal_packed = make_normal32(light_normal);
        const vec3 target_from_light = target_position - light_position;
        const vec3 connection_dir = normalize(target_from_light);
        const float facing_term = dot(connection_dir, light_normal);
        illuminant_tint = (facing_term > 0.f) ? buf.params.illuminant_tint : vec3(0.f);

        const float distance_sq = dot(target_from_light, target_from_light);
        const float area_ws = 4.f * PI * buf.params.radius_ws * buf.params.radius_ws;
        solid_angle_pdf_and_ext_bit = solid_angle_pdf_from_area_pdf(1.f/area_ws, facing_term, distance_sq);

        unit_scale = buf.params.unit_scale;
    }    
}
