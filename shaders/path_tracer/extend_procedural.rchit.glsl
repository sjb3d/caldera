#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "path_trace_common.glsl"

layout(shaderRecordEXT, scalar) buffer ExtendProceduralHitRecord {
    ProceduralHitRecordHeader header;
    // intersection data follows
} g_record;

hitAttributeEXT ProceduralHitAttribute g_attrib;

EXTEND_PAYLOAD_IN(g_extend);

void main()
{
    const bool is_front_hit = (gl_HitKindEXT == PROCEDURAL_HIT_FRONT);
    const vec3 front_normal_vec = get_vec(g_attrib.geom_normal);
    const vec3 hit_normal_vec_ls = is_front_hit ? front_normal_vec : -front_normal_vec;

    // transform normal vector to world space
    const vec3 hit_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_normal_vec_ls, 0.f);
    const vec3 hit_pos_ws = gl_WorldRayOriginEXT + gl_HitTEXT*gl_WorldRayDirectionEXT;

    // show the side of the hit
    uint bsdf_type = get_bsdf_type(g_record.header.shader);
    vec3 reflectance = g_record.header.shader.reflectance;
    if ((g_path_trace.flags & PATH_TRACE_FLAG_CHECK_HIT_FACE) != 0) {
        bsdf_type = BSDF_TYPE_DIFFUSE;
        reflectance = is_front_hit ? HIT_FRONT_COLOR : HIT_BACK_COLOR;
    }

    g_extend.info = create_hit_info(
        get_bsdf_type(g_record.header.shader),
        is_emissive(g_record.header.shader),
        g_record.header.shader.light_index,
        g_attrib.epsilon_exponent);
    g_extend.position_or_extdir = hit_pos_ws;
    g_extend.geom_normal = make_normal32(hit_normal_vec_ws);
    g_extend.shading_normal = g_extend.geom_normal;
    g_extend.bsdf_params = create_bsdf_params(
        reflectance,
        g_record.header.shader.roughness,
        get_material_index(g_record.header.shader),
        is_front_hit);
    g_extend.primitive_index = gl_PrimitiveID;
}
