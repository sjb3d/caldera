#define HIT_SHADER_COUNT_PER_INSTANCE       2

struct HitData {
    uvec2 bits;
};

#define HIT_DATA_Y_MAX_EXP_SHIFT    24
#define HIT_DATA_Y_MAX_EXP_MASK     0xff000000
#define HIT_DATA_Y_HAS_SURFACE_BIT  0x00010000
#define HIT_DATA_Y_HAS_LIGHT_BIT    0x00020000

HitData create_hit_data(
    vec3 reflectance,
    bool is_emissive,
    float max_position_value)
{
    int max_exponent_tmp = 0;
    frexp(max_position_value, max_exponent_tmp);
    const uint max_exponent = uint(max_exponent_tmp + 128);

    HitData hit;
    hit.bits.x = packHalf2x16(reflectance.xy);
    hit.bits.y
        = packHalf2x16(vec2(reflectance.z, 0.f))
        | HIT_DATA_Y_HAS_SURFACE_BIT
        | (is_emissive ? HIT_DATA_Y_HAS_LIGHT_BIT : 0)
        | (max_exponent << HIT_DATA_Y_MAX_EXP_SHIFT)
        ;
    return hit;
}
HitData create_miss_data()
{
    HitData hit;
    hit.bits = uvec2(0);
    return hit;
}

vec3 get_reflectance(HitData hit)
{
    return vec3(
        unpackHalf2x16(hit.bits.x).xy,
        unpackHalf2x16(hit.bits.y).x);
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

#define EXTEND_HIT_SHADER_OFFSET        0
#define EXTEND_MISS_SHADER_OFFSET       0
#define EXTEND_PAYLOAD_INDEX            0
#define EXTEND_PAYLOAD_READ(NAME)       layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_WRITE(NAME)      layout(location = 0) rayPayloadInEXT ExtendPayload NAME

struct OcclusionPayload {
    uint is_occluded;
};

#define OCCLUSION_HIT_SHADER_OFFSET     1
#define OCCLUSION_MISS_SHADER_OFFSET    1
#define OCCLUSION_PAYLOAD_INDEX         1
#define OCCLUSION_PAYLOAD_READ(NAME)    layout(location = 1) rayPayloadEXT OcclusionPayload NAME
#define OCCLUSION_PAYLOAD_WRITE(NAME)   layout(location = 1) rayPayloadInEXT OcclusionPayload NAME
