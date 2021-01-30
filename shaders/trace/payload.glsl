#define HIT_SHADER_COUNT_PER_INSTANCE       2

struct ExtendPayload {
    vec3 position;
    uint normal_oct32;
    uvec3 reflectance_and_emission; // half3 and half3
    uint flags_packed;
};

#define EXTEND_FLAGS_MAX_EXP_MASK   0x000000ff
#define EXTEND_FLAGS_VALID_BIT      0x00000800

#define EXTEND_HIT_SHADER_OFFSET    0
#define EXTEND_MISS_SHADER_OFFSET   0
#define EXTEND_PAYLOAD_INDEX        0
#define EXTEND_PAYLOAD_READ(NAME)   layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_WRITE(NAME)  layout(location = 0) rayPayloadInEXT ExtendPayload NAME

struct OcclusionPayload {
    uint is_occluded;
};

#define OCCLUSION_HIT_SHADER_OFFSET     1
#define OCCLUSION_MISS_SHADER_OFFSET    1
#define OCCLUSION_PAYLOAD_INDEX         1
#define OCCLUSION_PAYLOAD_READ(NAME)    layout(location = 1) rayPayloadEXT OcclusionPayload NAME
#define OCCLUSION_PAYLOAD_WRITE(NAME)   layout(location = 1) rayPayloadInEXT OcclusionPayload NAME
