#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "disc_common.glsl"

layout(shaderRecordEXT, scalar) buffer IntersectSphereRecord {
    DiscGeomData geom;
} g_record;

hitAttributeEXT DiscHitAttribute g_attrib;

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.geom.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const vec3 n = g_record.geom.normal;
    const float r = g_record.geom.radius;

    const float d_dot_n = dot(d, n);
    const float t = -dot(p, n)/d_dot_n;
    const vec3 hit_from_centre = p + t*d;
    if (dot(hit_from_centre, hit_from_centre) < r*r) {
        g_attrib.hit_from_centre = hit_from_centre - hit_from_centre*dot(hit_from_centre, n);
        reportIntersectionEXT(t, sign_bit_set(d_dot_n) ? DISC_HIT_FRONT : DISC_HIT_BACK);
    }
}
