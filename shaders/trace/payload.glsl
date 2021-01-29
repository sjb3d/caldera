
struct ExtendPayload {
    uint packed_normal; // oct format
    uint packed_shader; // BSDF + parameters
};

#define EXTEND_PAYLOAD_READ(NAME)   layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_WRITE(NAME)  layout(location = 0) rayPayloadInEXT ExtendPayload NAME
