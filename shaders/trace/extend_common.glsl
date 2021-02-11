struct HitData {
    uvec3 bits;
};

#define HIT_DATA_Z_LIGHT_INDEX_MASK     0x0000ffff
#define HIT_DATA_Z_HAS_SURFACE_BIT      0x00010000
#define HIT_DATA_Z_HAS_LIGHT_BIT        0x00020000
#define HIT_DATA_Z_BSDF_TYPE_SHIFT      20
#define HIT_DATA_Z_BSDF_TYPE_MASK       0x00300000
#define HIT_DATA_Z_EPSILON_EXP_SHIFT    24
#define HIT_DATA_Z_EPSILON_EXP_MASK     0xff000000

#define BSDF_TYPE_DIFFUSE       0
#define BSDF_TYPE_GGX           1
#define BSDF_TYPE_MIRROR        2

HitData create_hit_data(
    uint bsdf_type,
    vec3 reflectance,
    float roughness,
    bool is_emissive,
    uint light_index,
    float epsilon_ref)
{
    int exponent;
    frexp(epsilon_ref, exponent);
    const uint biased_exponent = uint(exponent + 128);

    HitData hit;
    hit.bits.x = packHalf2x16(reflectance.xy);
    hit.bits.y = packHalf2x16(vec2(reflectance.z, roughness));
    hit.bits.z
        = light_index
        | HIT_DATA_Z_HAS_SURFACE_BIT
        | (is_emissive ? HIT_DATA_Z_HAS_LIGHT_BIT : 0)
        | (bsdf_type << HIT_DATA_Z_BSDF_TYPE_SHIFT)
        | (biased_exponent << HIT_DATA_Z_EPSILON_EXP_SHIFT)
        ;
    return hit;
}
HitData create_miss_data()
{
    HitData hit;
    hit.bits = uvec3(0);
    return hit;
}

HitData replace_hit_data(HitData hit, uint bsdf_type, float roughness)
{
    hit.bits.y &= 0x0000ffff;
    hit.bits.y |= packHalf2x16(vec2(0.f, roughness));
    
    hit.bits.z &= ~HIT_DATA_Z_BSDF_TYPE_MASK;
    hit.bits.z |= (bsdf_type << HIT_DATA_Z_BSDF_TYPE_SHIFT);

    return hit;
}

uint get_bsdf_type(HitData hit)
{
    return (hit.bits.z & HIT_DATA_Z_BSDF_TYPE_MASK) >> HIT_DATA_Z_BSDF_TYPE_SHIFT;
}
vec3 get_reflectance(HitData hit)
{
    return vec3(
        unpackHalf2x16(hit.bits.x),
        unpackHalf2x16(hit.bits.y).x);
}
float get_roughness(HitData hit)
{
    return unpackHalf2x16(hit.bits.y).y;
}
uint get_light_index(HitData hit)
{
    return hit.bits.z & HIT_DATA_Z_LIGHT_INDEX_MASK;
}
bool has_light(HitData hit)
{
    return (hit.bits.z & HIT_DATA_Z_HAS_LIGHT_BIT) != 0;
}
bool has_surface(HitData hit)
{
    return (hit.bits.z & HIT_DATA_Z_HAS_SURFACE_BIT) != 0;
}
float get_epsilon(HitData hit, int exponent_offset_from_ref)
{
    const int exponent = int(hit.bits.z >> HIT_DATA_Z_EPSILON_EXP_SHIFT) - 128;
    return ldexp(1.f, exponent + exponent_offset_from_ref);
}

struct ExtendPayload {
    vec3 position_or_extdir;
    uint normal_oct32;
    HitData hit;
};

struct ExtendShader {
    uint flags;
    vec3 reflectance;
    float roughness;
    uint light_index;
};

#define HIT_SHADER_COUNT_PER_INSTANCE       2

#define EXTEND_HIT_SHADER_OFFSET        0
#define EXTEND_MISS_SHADER_OFFSET       0
#define EXTEND_PAYLOAD_INDEX            0
#define EXTEND_PAYLOAD(NAME)            layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_IN(NAME)         layout(location = 0) rayPayloadInEXT ExtendPayload NAME

#define EXTEND_SHADER_FLAGS_BSDF_TYPE_MASK      0x3
#define EXTEND_SHADER_FLAGS_IS_EMISSIVE_BIT     0x4
