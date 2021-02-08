#define SPHERE_HIT_RECORD(NAME)                                 \
    layout(shaderRecordEXT, scalar) buffer SphereHitRecord {    \
        vec3 centre;                                            \
        float radius;                                           \
        vec3 reflectance;                                       \
        uint flags;                                             \
    } NAME

struct SphereHitAttribute {
    vec3 hit_from_centre;
};
