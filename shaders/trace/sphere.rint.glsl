#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "sphere_common.glsl"

layout(shaderRecordEXT, scalar) buffer IntersectSphereRecord {
    SphereGeomData geom;
} g_record;

hitAttributeEXT SphereHitAttribute g_attrib;

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.geom.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const float r = g_record.geom.radius;

    vec2 t;
    if (ray_vs_sphere(p, d, r, t)) {
        const float t_min = min_element(t);
        const float t_max = max_element(t);
        g_attrib.hit_from_centre = p + t_min*d;
        if (!reportIntersectionEXT(t_min, SPHERE_HIT_FRONT)) {
            g_attrib.hit_from_centre = p + t_max*d;
            reportIntersectionEXT(t_max, SPHERE_HIT_BACK);
        }
    }
}
