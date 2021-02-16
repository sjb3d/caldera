#ifndef INCLUDED_COMMON_NORMAL_PACK
#define INCLUDED_COMMON_NORMAL_PACK

#include "maths.glsl"

// reference: "Survey of Efficient Representations for Independent Unit Vectors"
// http://jcgt.org/published/0003/02/01/
vec2 oct_from_vec(vec3 v)
{
    const vec2 p = v.xy/sum_elements(abs(v));
    return (v.z > 0.f) ? p : copysign(1.f - abs(p.yx), p);
}
vec3 vec_from_oct(vec2 p)
{
    const float z = 1.f - sum_elements(abs(p));
    const vec2 xy = (z > 0.f) ? p : copysign(1.f - abs(p.yx), p);
    return vec3(xy, z);
}

uint oct32_from_vec(vec3 v)
{
     return packSnorm2x16(oct_from_vec(v));
}
vec3 vec_from_oct32(uint p)
{
    return vec_from_oct(unpackSnorm2x16(p));
}

struct Normal32 {
    uint bits;
};

Normal32 make_normal32(vec3 vec) 
{
    Normal32 n;
    n.bits = oct32_from_vec(vec);
    return n;
}
vec3 get_vec(Normal32 n) { return vec_from_oct32(n.bits); }
vec3 get_dir(Normal32 n) { return normalize(get_vec(n)); }

#endif
