struct OcclusionPayload {
    uint is_occluded;
};

#define OCCLUSION_HIT_SHADER_OFFSET     1
#define OCCLUSION_MISS_SHADER_OFFSET    1
#define OCCLUSION_PAYLOAD_INDEX         1
#define OCCLUSION_PAYLOAD(NAME)         layout(location = 1) rayPayloadEXT OcclusionPayload NAME
#define OCCLUSION_PAYLOAD_IN(NAME)      layout(location = 1) rayPayloadInEXT OcclusionPayload NAME
