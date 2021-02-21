struct DiscGeomData {
    vec3 centre;
    vec3 normal;
    float radius;
};

struct DiscHitAttribute {
    vec3 hit_from_centre;
};

#define DISC_HIT_FRONT      1
#define DISC_HIT_BACK       0
