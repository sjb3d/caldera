#ifndef INCLUDED_EXTEND_COMMON
#define INCLUDED_EXTEND_COMMON

#include "bsdf_common.glsl"
#include "normal_pack.glsl"

struct HitInfo {
    uint bits;
};

#define HIT_INFO_LIGHT_INDEX_MASK       0x0000ffff
#define HIT_INFO_HAS_SURFACE_BIT        0x00010000
#define HIT_INFO_HAS_LIGHT_BIT          0x00020000
#define HIT_INFO_BSDF_TYPE_SHIFT        20
#define HIT_INFO_BSDF_TYPE_MASK         0x00f00000
#define HIT_INFO_UNIT_SCALE_EXP_SHIFT   24
#define HIT_INFO_UNIT_SCALE_EXP_MASK    0xff000000

HitInfo create_hit_info(
    uint bsdf_type,
    bool is_emissive,
    uint light_index,
    float unit_scale)
{
    int exponent;
    frexp(unit_scale, exponent);
    const uint biased_exponent = uint(exponent + 128);

    HitInfo hit;
    hit.bits
        = light_index
        | HIT_INFO_HAS_SURFACE_BIT
        | (is_emissive ? HIT_INFO_HAS_LIGHT_BIT : 0)
        | (bsdf_type << HIT_INFO_BSDF_TYPE_SHIFT)
        | (biased_exponent << HIT_INFO_UNIT_SCALE_EXP_SHIFT)
        ;
    return hit;
}
HitInfo create_miss_info()
{
    HitInfo hit;
    hit.bits = 0;
    return hit;
}

HitInfo replace_bsdf_type(HitInfo hit, uint bsdf_type)
{
    hit.bits &= ~HIT_INFO_BSDF_TYPE_MASK;
    hit.bits |= (bsdf_type << HIT_INFO_BSDF_TYPE_SHIFT);
    return hit;
}

uint get_bsdf_type(HitInfo hit)
{
    return (hit.bits & HIT_INFO_BSDF_TYPE_MASK) >> HIT_INFO_BSDF_TYPE_SHIFT;
}
uint get_light_index(HitInfo hit)
{
    return hit.bits & HIT_INFO_LIGHT_INDEX_MASK;
}
bool has_light(HitInfo hit)
{
    return (hit.bits & HIT_INFO_HAS_LIGHT_BIT) != 0;
}
bool has_surface(HitInfo hit)
{
    return (hit.bits & HIT_INFO_HAS_SURFACE_BIT) != 0;
}
float get_epsilon(HitInfo hit, int exponent_offset_from_unit_scale)
{
    const int exponent = int(hit.bits >> HIT_INFO_UNIT_SCALE_EXP_SHIFT) - 128;
    return ldexp(1.f, exponent + exponent_offset_from_unit_scale);
}

struct ExtendPayload {
    HitInfo info;
    vec3 position_or_extdir;
    Normal32 geom_normal;
    Normal32 shading_normal;
    BsdfParams bsdf_params;
    uint primitive_index;
};

struct ExtendShader {
    uint flags;
    vec3 reflectance;
    float roughness;
    uint light_index;
};

#define HIT_SHADER_COUNT_PER_INSTANCE   2

#define EXTEND_HIT_SHADER_OFFSET        0
#define EXTEND_MISS_SHADER_OFFSET       0
#define EXTEND_PAYLOAD_INDEX            0
#define EXTEND_PAYLOAD(NAME)            layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_IN(NAME)         layout(location = 0) rayPayloadInEXT ExtendPayload NAME

#define EXTEND_SHADER_FLAGS_TEXTURE_INDEX_MASK  0x0000ffff
#define EXTEND_SHADER_FLAGS_BSDF_TYPE_MASK      0x000f0000
#define EXTEND_SHADER_FLAGS_MATERIAL_INDEX_MASK 0x00f00000
#define EXTEND_SHADER_FLAGS_HAS_NORMALS_BIT     0x01000000
#define EXTEND_SHADER_FLAGS_HAS_TEXTURE_BIT     0x02000000
#define EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT     0x04000000
#define EXTEND_SHADER_FLAGS_IS_CHECKERBOARD_BIT 0x08000000

uint get_texture_index(ExtendShader s)
{
    return s.flags & EXTEND_SHADER_FLAGS_TEXTURE_INDEX_MASK;
}
uint get_bsdf_type(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_BSDF_TYPE_MASK) >> 16;
}
uint get_material_index(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_MATERIAL_INDEX_MASK) >> 20;
}
bool is_emissive(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT) != 0;
}
bool has_normals(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_HAS_NORMALS_BIT) != 0;
}
bool has_texture(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_HAS_TEXTURE_BIT) != 0;
}
bool is_checkerboard(ExtendShader s)
{
    return (s.flags & EXTEND_SHADER_FLAGS_IS_CHECKERBOARD_BIT) != 0;
}

#define PROCEDURAL_HIT_FRONT    1
#define PROCEDURAL_HIT_BACK     0

struct ProceduralHitRecordHeader {
    float unit_scale;
    ExtendShader shader;
};

struct ProceduralHitAttribute {
    Normal32 geom_normal;
};

#endif
