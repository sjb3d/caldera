#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"

layout(shaderRecordEXT, scalar) buffer IntersectSphereRecord {
    ProceduralHitRecordHeader header;
    vec3 centre;
    vec3 normal;
    float radius;
} g_record;

hitAttributeEXT ProceduralHitAttribute g_attrib;

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const vec3 n = g_record.normal;
    const float r = g_record.radius;

    const float d_dot_n = dot(d, n);
    const float t = -dot(p, n)/d_dot_n;
    const vec3 hit_from_centre = p + t*d;
    if (dot(hit_from_centre, hit_from_centre) < r*r) {
        g_attrib.geom_normal = make_normal32(n);
        g_attrib.epsilon_exponent = default_epsilon_exponent(g_record.header.unit_scale);
        reportIntersectionEXT(t, sign_bit_set(d_dot_n) ? PROCEDURAL_HIT_FRONT : PROCEDURAL_HIT_BACK);
    }
}
