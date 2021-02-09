#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sampler.glsl"
#include "light_common.glsl"

LIGHT_UNIFORM_DATA(g_light);

SPHERE_LIGHT_RECORD(g_record);

LIGHT_SAMPLE_DATA_IN(g_sample);

void main()
{
    const vec3 target_position = g_sample.position;
    //const vec3 target_normal = g_sample.normal;
    const vec2 rand_u01 = g_sample.emission.xy;

    if (g_light.sample_sphere_solid_angle != 0) {
        const vec3 centre_from_target = g_record.centre_ws - target_position;

        const float sin_theta = g_record.radius_ws/length(centre_from_target);
        const float cos_theta = sqrt(max(0.f, 1.f - sin_theta*sin_theta));
        const float solid_angle = 2.f*PI*(1.f - cos_theta);

        const vec3 ray_dir_ls = sample_solid_angle_uniform(cos_theta, rand_u01);
        const mat3 basis = basis_from_z_axis(normalize(centre_from_target));
        const vec3 ray_dir = basis*ray_dir_ls;

        // intersect ray with sphere for sample point
        const vec3 p = -centre_from_target;
        const vec3 d = ray_dir;
        const float r = g_record.radius_ws;

        const float a = dot(d, d);
        const float b = 2.f*dot(d, p);
        const float c = dot(p, p) - r*r;

        const float Q = max(b*b - 4.f*a*c, 0.f);
        const float q = sqrt(Q);
        const float k = -b - copysign(q, b);
        const float t1 = k/(2.f*a);
        const float t2 = (2.f*c)/k;
        const float t_min = max(min(t1, t2), 0.f);

        // construct the sample here
        const vec3 sample_dir = normalize(p + t_min*d);
        const vec3 light_position = g_record.centre_ws + g_record.radius_ws*sample_dir;
        const vec3 light_normal = sample_dir;
        const vec3 target_from_light = target_position - light_position;
        const float facing_term = dot(target_from_light, light_normal);
        const vec3 emission = (facing_term > 0.f) ? g_record.emission : vec3(0.f);

        g_sample.position = light_position;
        g_sample.normal = light_normal;
        g_sample.emission = emission;
        g_sample.solid_angle_pdf = 1.f/solid_angle;
        g_sample.unit_value = g_record.unit_value;
    } else {
        const vec3 sample_dir = sample_sphere_uniform(rand_u01);
        const vec3 light_position = g_record.centre_ws + g_record.radius_ws*sample_dir;
        const vec3 light_normal = sample_dir;
        const vec3 target_from_light = target_position - light_position;
        const vec3 connection_dir = normalize(target_from_light);
        const float facing_term = dot(connection_dir, light_normal);
        const vec3 emission = (facing_term > 0.f) ? g_record.emission : vec3(0.f);

        const float distance_sq = dot(target_from_light, target_from_light);
        const float area_ws = 4.f * PI * g_record.radius_ws * g_record.radius_ws;
        const float solid_angle_pdf = solid_angle_pdf_from_area_pdf(1.f/area_ws, facing_term, distance_sq);

        g_sample.position = light_position;
        g_sample.normal = light_normal;
        g_sample.emission = emission;
        g_sample.solid_angle_pdf = solid_angle_pdf;
        g_sample.unit_value = g_record.unit_value;   
    }
}
