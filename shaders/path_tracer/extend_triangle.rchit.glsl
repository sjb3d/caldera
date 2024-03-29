#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#extension GL_GOOGLE_include_directive : require
#include "maths.glsl"
#include "extend_common.glsl"
#include "path_trace_common.glsl"

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 tri[];
};

layout(buffer_reference, scalar) buffer PositionBuffer {
    vec3 pos[];
};
layout(buffer_reference, scalar) buffer NormalBuffer {
    vec3 normal[];
};
layout(buffer_reference, scalar) buffer UvBuffer {
    vec2 uv[];
};

layout(shaderRecordEXT, scalar) buffer ExtendTriangleHitRecord {
    IndexBuffer index_buffer;
    PositionBuffer position_buffer;
    NormalBuffer normal_buffer;
    UvBuffer uv_buffer;
    float unit_scale;
    ExtendShader shader;
    uint pad;
} g_record;

hitAttributeEXT vec2 g_bary_coord;

EXTEND_PAYLOAD_IN(g_extend);

void main()
{   
    const bool is_front_hit = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);

    const uvec3 tri = g_record.index_buffer.tri[gl_PrimitiveID];
    const vec3 p0 = g_record.position_buffer.pos[tri[0]];
    const vec3 p1 = g_record.position_buffer.pos[tri[1]];
    const vec3 p2 = g_record.position_buffer.pos[tri[2]];

    const vec3 geom_normal_vec_ls = cross(p2 - p1, p0 - p1);
    const vec3 hit_pos_ls
        = p0*(1.f - g_bary_coord.x - g_bary_coord.y)
        + p1*g_bary_coord.x
        + p2*g_bary_coord.y
        ;

    vec3 shading_normal_vec_ls = geom_normal_vec_ls;
    if (has_normals(g_record.shader)) {
        const vec3 n0 = g_record.normal_buffer.normal[tri[0]];
        const vec3 n1 = g_record.normal_buffer.normal[tri[1]];
        const vec3 n2 = g_record.normal_buffer.normal[tri[2]];

        shading_normal_vec_ls
            = n0*(1.f - g_bary_coord.x - g_bary_coord.y)
            + n1*g_bary_coord.x
            + n2*g_bary_coord.y
            ;         
    }

    vec3 reflectance;
    if (has_texture(g_record.shader)) {
        const vec2 uv0 = g_record.uv_buffer.uv[tri[0]];
        const vec2 uv1 = g_record.uv_buffer.uv[tri[1]];
        const vec2 uv2 = g_record.uv_buffer.uv[tri[2]];
        const vec2 uv
            = uv0*(1.f - g_bary_coord.x - g_bary_coord.y)
            + uv1*g_bary_coord.x
            + uv2*g_bary_coord.y
            ;

        const uint texture_index = get_texture_index(g_record.shader);
        reflectance = texture(sampler2D(g_textures[nonuniformEXT(texture_index)], g_linear_sampler), uv).xyz;        
    } else {
        reflectance = g_record.shader.reflectance;
    }
    
    if (is_checkerboard(g_record.shader)) {
        const vec3 coord_flt = hit_pos_ls*2.f;
        const uvec3 coord = uvec3(ivec3(coord_flt)) ^ (floatBitsToUint(coord_flt) >> 31);
        const uint p = coord.x ^ coord.y ^ coord.z;
        if ((p & 1) == 0) {
            reflectance = .66f*reflectance;
        }
    }

    // transform to world space
    const vec3 hit_geom_normal_vec_ls = is_front_hit ? geom_normal_vec_ls : -geom_normal_vec_ls;
    const vec3 hit_shading_normal_vec_ls = is_front_hit ? shading_normal_vec_ls : -shading_normal_vec_ls;
    const vec3 hit_geom_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_geom_normal_vec_ls, 0.f);
    const vec3 hit_shading_normal_vec_ws = gl_ObjectToWorldEXT * vec4(hit_shading_normal_vec_ls, 0.f);
    const vec3 hit_pos_ws = gl_ObjectToWorldEXT * vec4(hit_pos_ls, 1.f);

    // show the side of the hit
    uint bsdf_type = get_bsdf_type(g_record.shader);
    if ((g_path_trace.flags & PATH_TRACE_FLAG_CHECK_HIT_FACE) != 0) {
        bsdf_type = BSDF_TYPE_DIFFUSE;
        reflectance = is_front_hit ? HIT_FRONT_COLOR : HIT_BACK_COLOR;
    }

    g_extend.info = create_hit_info(
        bsdf_type,
        is_emissive(g_record.shader),
        g_record.shader.light_index,
        default_epsilon_exponent(g_record.unit_scale));
    g_extend.position_or_extdir = hit_pos_ws;
    g_extend.geom_normal = make_normal32(hit_geom_normal_vec_ws);
    g_extend.shading_normal = make_normal32(hit_shading_normal_vec_ws);
    g_extend.bsdf_params = create_bsdf_params(
        reflectance,
        g_record.shader.roughness,
        get_material_index(g_record.shader),
        is_front_hit);
    g_extend.primitive_index = gl_PrimitiveID;
}
