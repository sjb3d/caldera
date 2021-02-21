#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "disc_common.glsl"

layout(shaderRecordEXT, scalar) buffer ExtendDiscHitRecord {
    DiscGeomData geom;
    float unit_scale;
    ExtendShader shader;
} g_record;

hitAttributeEXT DiscHitAttribute g_attrib;

EXTEND_PAYLOAD_IN(g_extend);

void main()
{   
    const vec3 hit_pos_ls = g_record.geom.centre + g_attrib.hit_from_centre;
    const bool is_front_hit = (gl_HitKindEXT == DISC_HIT_FRONT);
    const vec3 hit_normal_vec_ls = is_front_hit ? g_record.geom.normal : -g_record.geom.normal;

    // transform normal vector to world space
    const vec3 hit_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_normal_vec_ls, 0.f);
    const vec3 hit_pos_ws = gl_ObjectToWorldEXT * vec4(hit_pos_ls, 1.f);

    g_extend.info = create_hit_info(
        get_bsdf_type(g_record.shader),
        is_emissive(g_record.shader),
        g_record.shader.light_index,
        g_record.unit_scale);
    g_extend.position_or_extdir = hit_pos_ws;
    g_extend.geom_normal = make_normal32(hit_normal_vec_ws);
    g_extend.shading_normal = g_extend.geom_normal;
    g_extend.bsdf_params = create_bsdf_params(
        g_record.shader.reflectance,
        g_record.shader.roughness,
        is_front_hit);
}