#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "normal_pack.glsl"
#include "sphere_common.glsl"

layout(shaderRecordEXT, scalar) buffer ExtendSphereHitRecord {
    SphereGeomData geom;
    ExtendShader shader;
} g_sphere;

hitAttributeEXT SphereHitAttribute g_attrib;

EXTEND_PAYLOAD_IN(g_extend);

void main()
{   
    const vec3 hit_from_centre = g_attrib.hit_from_centre;
    const vec3 hit_pos_ls = g_sphere.geom.centre + hit_from_centre;
    const vec3 hit_normal_vec_ls = (gl_HitKindEXT != 0) ? hit_from_centre : -hit_from_centre;

    // transform normal vector to world space
    const vec3 hit_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_normal_vec_ls, 0.f);
    const vec3 hit_pos_ws = gl_ObjectToWorldEXT * vec4(hit_pos_ls, 1.f);

    const uint bsdf_type = g_sphere.shader.flags & EXTEND_SHADER_FLAGS_BSDF_TYPE_MASK;
    const bool is_emissive = ((g_sphere.shader.flags & EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT) != 0);

    // estimate floating point number size for local and world space
    // TODO: handle scale
    const float max_position_value = max_element(
        abs(g_sphere.geom.centre)
        + abs(gl_ObjectToWorldEXT[3])
    ) + abs(g_sphere.geom.radius);
    int max_exponent = 0;
    frexp(max_position_value, max_exponent);

    g_extend.position = hit_pos_ws;
    g_extend.normal_oct32 = oct32_from_vec(hit_normal_vec_ws);
    g_extend.hit = create_hit_data(
        bsdf_type,
        g_sphere.shader.reflectance,
        is_emissive,
        g_sphere.shader.light_index,
        max_exponent);
}
