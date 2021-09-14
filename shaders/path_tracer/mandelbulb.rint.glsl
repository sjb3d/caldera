#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "path_trace_common.glsl"

layout(shaderRecordEXT, scalar) buffer IntersectMandelbulbRecord {
    ProceduralHitRecordHeader header;
    vec3 centre;
} g_record;

hitAttributeEXT ProceduralHitAttribute g_attrib;

vec3 sph_pow8(vec3 p, float r, inout float dr)
{
    // use expansion from iq to take to the power of 8 without trig
    // reference: https://www.shadertoy.com/view/ltfSWn
    const float r2 = r*r;
    const float r4 = r2*r2;
    dr = 8.f*(r*r2*r4)*dr + 1.f;

    const vec3 p2 = p*p;
    const vec3 p4 = p2*p2;
    const float x = p.x; const float x2 = p2.x; const float x4 = p4.x;
    const float y = p.y; const float y2 = p2.y; const float y4 = p4.y;
    const float z = p.z; const float z2 = p2.z; const float z4 = p4.z;

    const float k3 = x2 + z2;
    const float k2 = inversesqrt(k3*k3*k3*k3*k3*k3*k3);
    const float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
    const float k4 = x2 - y2 + z2;

    return vec3(
        64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2,
        -16.0*y2*k3*k4*k4 + k1*k1,
        -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2
    );
}

#define BULB_ITERATIONS             8
#define BULB_RADIUS                 1.1f
#define BULB_MAX_STEPS              512
#define BULB_EPSILON_IN_PIXELS      .1f

float bulb_distance(vec3 pos)
{
    vec3 p = pos;
    float r = length(p);
    float dr = 1.f;

    for (int i = 0; i < BULB_ITERATIONS && r < 16.f; ++i) {
        p = sph_pow8(p, r, dr) + pos;
        r = length(p);
    }

    return .5f*log(r)*r/dr;
}

vec3 bulb_grad(vec3 pos, float epsilon)
{
    const vec2 e = vec2(-epsilon, epsilon);
    return flip_sign(vec3(bulb_distance(pos + e.yxx)), e.yxx)
         + flip_sign(vec3(bulb_distance(pos + e.xyx)), e.xyx)
         + flip_sign(vec3(bulb_distance(pos + e.xxy)), e.xxy)
         + flip_sign(vec3(bulb_distance(pos + e.yyy)), e.yyy)
         ;
}

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const float r = BULB_RADIUS;

    const vec3 camera_pos = gl_WorldToObjectEXT * vec4(g_path_trace.camera.world_from_local[3], 1.f);

    vec2 sphere_t;
    if (ray_vs_sphere(p, d, r, sphere_t)) {
        const float t_min = max(min_element(sphere_t), gl_RayTminEXT);
        const float t_max = min(max_element(sphere_t), gl_RayTmaxEXT);

        const float d_len_rcp = inversesqrt(dot(d, d));
        float t = t_min;
        float prev_t = t;
        float prev_dist = 0.f;
        uint step_index = 0;
        while (t < t_max && step_index < BULB_MAX_STEPS) {
            const vec3 pos = p + t*d;
            const float dist = bulb_distance(p + t*d);
            const float epsilon = BULB_EPSILON_IN_PIXELS*length(camera_pos - pos)*g_path_trace.camera.pixel_size_at_unit_z;
            if (dist < epsilon) {
                t = mix(prev_t, t, unlerp(prev_dist, dist, epsilon));
                g_attrib.geom_normal = make_normal32(bulb_grad(p + t*d, .5f*epsilon));
                int epsilon_exponent;
                frexp(epsilon, epsilon_exponent);
                g_attrib.epsilon_exponent = epsilon_exponent;
                reportIntersectionEXT(t, PROCEDURAL_HIT_FRONT);
                break;
            }
            prev_t = t;
            prev_dist = dist;
            t += dist*d_len_rcp;
            ++step_index;
        }
    }
}
