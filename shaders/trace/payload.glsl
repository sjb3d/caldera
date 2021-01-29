
struct ExtendPayload {
    vec3 position;
    uint normal_oct32;
    uvec3 reflectance_and_emission; // half3 and half3
    uint is_valid;
};

#define EXTEND_PAYLOAD_READ(NAME)   layout(location = 0) rayPayloadEXT ExtendPayload NAME
#define EXTEND_PAYLOAD_WRITE(NAME)  layout(location = 0) rayPayloadInEXT ExtendPayload NAME
