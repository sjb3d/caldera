struct SphereGeomData {
    vec3 centre;
    float radius;
};

struct SphereHitAttribute {
    vec3 hit_from_centre;
};

#define SPHERE_HIT_FRONT    1
#define SPHERE_HIT_BACK     0
