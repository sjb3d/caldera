#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sphere_common.glsl"

SPHERE_HIT_RECORD(g_record);

hitAttributeEXT SphereHitAttribute g_attrib;

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const float r = g_record.radius;

    /*
        |p + t*d|^2 = r^2
        => d.d t^2 + 2(d.p)t + (p.p - r^2) = 0
    */
    const float a = dot(d, d);
    const float b = 2.f*dot(d, p);
    const float c = dot(p, p) - r*r;

    const float Q = b*b - 4.f*a*c;
    if (Q > 0.f) {
        const float q = sqrt(Q);
        const float k = -b - copysign(q, b);
        const float t1 = k/(2.f*a);
        const float t2 = (2.f*c)/k;
        const float t_min = min(t1, t2);
        const float t_max = max(t1, t2);
        g_attrib.hit_from_centre = p + t_min*d;
        if (!reportIntersectionEXT(t_min, 1)) {
            g_attrib.hit_from_centre = p + t_max*d;
            reportIntersectionEXT(t_max, 0);
        }
    }
}
