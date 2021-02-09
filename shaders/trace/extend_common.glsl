struct HitData {
    uvec3 bits;
};

#define HIT_DATA_Y_MAX_EXP_SHIFT        24
#define HIT_DATA_Y_MAX_EXP_MASK         0xff000000
#define HIT_DATA_Y_HAS_SURFACE_BIT      0x00010000
#define HIT_DATA_Y_HAS_LIGHT_BIT        0x00020000
#define HIT_DATA_Y_BSDF_TYPE_SHIFT      20
#define HIT_DATA_Y_BSDF_TYPE_MASK       0x00300000

#define BSDF_TYPE_DIFFUSE       0
#define BSDF_TYPE_GGX           1
#define BSDF_TYPE_MIRROR        2

HitData create_hit_data(
    uint bsdf_type,
    vec3 reflectance,
    bool is_emissive,
    uint light_index,
    int max_exponent)
{
    const uint biased_max_exponent = uint(max_exponent + 128);

    HitData hit;
    hit.bits.x = packHalf2x16(reflectance.xy);
    hit.bits.y
        = packHalf2x16(vec2(reflectance.z, 0.f))
        | HIT_DATA_Y_HAS_SURFACE_BIT
        | (is_emissive ? HIT_DATA_Y_HAS_LIGHT_BIT : 0)
        | (bsdf_type << HIT_DATA_Y_BSDF_TYPE_SHIFT)
        | (biased_max_exponent << HIT_DATA_Y_MAX_EXP_SHIFT)
        ;
    hit.bits.z = light_index;
    return hit;
}
HitData create_miss_data(bool is_emissive, uint light_index)
{
    HitData hit;
    hit.bits.x = 0;
    hit.bits.y = is_emissive ? HIT_DATA_Y_HAS_LIGHT_BIT : 0;
    hit.bits.z = light_index;
    return hit;
}

uint get_bsdf_type(HitData hit)
{
    return (hit.bits.y & HIT_DATA_Y_BSDF_TYPE_MASK) >> HIT_DATA_Y_BSDF_TYPE_SHIFT;
}
vec3 get_reflectance(HitData hit)
{
    return vec3(
        unpackHalf2x16(hit.bits.x).xy,
        unpackHalf2x16(hit.bits.y).x);
}
uint get_light_index(HitData hit)
{
    return hit.bits.z;
}
bool has_light(HitData hit)
{
    return (hit.bits.y & HIT_DATA_Y_HAS_LIGHT_BIT) != 0;
}
bool has_surface(HitData hit)
{
    return (hit.bits.y & HIT_DATA_Y_HAS_SURFACE_BIT) != 0;
}
int get_max_exponent(HitData hit)
{
    return int(hit.bits.y >> HIT_DATA_Y_MAX_EXP_SHIFT) - 128;
}

struct ExtendPayload {
    vec3 position;
    uint normal_oct32;
    HitData hit;
};

struct ExtendShader {
    uint flags;
    vec3 reflectance;
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
