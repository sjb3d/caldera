#include "light_common.glsl"

void solid_angle_light_eval(
    SolidAngleLightParams params,
    vec3 target_position,
    vec3 light_extdir,
    out vec3 emission,
    out float solid_angle_pdf)
{
    // check we are within the solid angle of the light
    const float cos_theta = dot(params.direction_ws, light_extdir);
    const float cos_theta_min = 1.f - params.solid_angle/(2.f*PI);
    emission = (cos_theta > cos_theta_min) ? params.emission : vec3(0.f);
    solid_angle_pdf = 1.f/params.solid_angle;
}

void solid_angle_light_sample(
    SolidAngleLightParams params,
    vec3 target_position,
    vec3 target_normal,
    vec2 rand_u01,
    out vec3 light_extdir,
    out Normal32 light_normal_packed,
    out vec3 emission,
    out float solid_angle_pdf_and_ext_bit,
    out float unit_scale)
{
    const float cos_theta = 1.f - params.solid_angle/(2.f*PI);
    const vec3 ray_dir_ls = sample_solid_angle_uniform(cos_theta, rand_u01);

    const mat3 basis = basis_from_z_axis(normalize(params.direction_ws));
    light_extdir = normalize(basis*ray_dir_ls);
    light_normal_packed = make_normal32(-light_extdir);

    emission = params.emission;
    solid_angle_pdf_and_ext_bit = -1.0/params.solid_angle;

    unit_scale = 0.f;
}
