#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"

layout(shaderRecordEXT, scalar) buffer IntersectSphereRecord {
    ProceduralHitRecordHeader header;
    vec3 centre;
    float radius;
} g_record;

hitAttributeEXT ProceduralHitAttribute g_attrib;

void main()
{
    const vec3 p = gl_ObjectRayOriginEXT - g_record.centre;
    const vec3 d = gl_ObjectRayDirectionEXT;
    const float r = g_record.radius;

    vec2 t;
    if (ray_vs_sphere(p, d, r, t)) {
        const float t_min = min_element(t);
        const float t_max = max_element(t);
        g_attrib.geom_normal = make_normal32(p + t_min*d);
        if (!reportIntersectionEXT(t_min, PROCEDURAL_HIT_FRONT)) {
            g_attrib.geom_normal = make_normal32(p + t_max*d);
            reportIntersectionEXT(t_max, PROCEDURAL_HIT_BACK);
        }
    }
}
