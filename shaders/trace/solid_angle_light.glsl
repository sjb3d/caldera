#include "light_common.glsl"

void solid_angle_light_eval(
    uint64_t params_addr,
    vec3 target_position,
    vec3 light_extdir,
    out vec3 illuminant_tint,
    out float solid_angle_pdf)
{
    SolidAngleLightParamsBuffer buf = SolidAngleLightParamsBuffer(params_addr);

    // check we are within the solid angle of the light
    const float cos_theta = dot(buf.params.direction_ws, light_extdir);
    const float cos_theta_min = 1.f - buf.params.solid_angle/(2.f*PI);
    illuminant_tint = (cos_theta > cos_theta_min) ? buf.params.illuminant_tint : vec3(0.f);
    solid_angle_pdf = 1.f/buf.params.solid_angle;
}

void solid_angle_light_sample(
    uint64_t params_addr,
    vec3 target_position,
    vec3 target_normal,
    vec2 light_rand_u01,
    out vec3 light_extdir,
    out Normal32 light_normal_packed,
    out vec3 illuminant_tint,
    out float solid_angle_pdf_and_ext_bit,
    out float unit_scale)
{
    SolidAngleLightParamsBuffer buf = SolidAngleLightParamsBuffer(params_addr);

    const float cos_theta = 1.f - buf.params.solid_angle/(2.f*PI);
    const vec3 ray_dir_ls = sample_solid_angle_uniform(cos_theta, light_rand_u01);

    const mat3 basis = basis_from_z_axis(normalize(buf.params.direction_ws));
    light_extdir = normalize(basis*ray_dir_ls);
    light_normal_packed = make_normal32(-light_extdir);

    illuminant_tint = buf.params.illuminant_tint;
    solid_angle_pdf_and_ext_bit = -1.0/buf.params.solid_angle;

    unit_scale = 0.f;
}
