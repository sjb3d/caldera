#ifndef INCLUDED_COMMON_BSDF
#define INCLUDED_COMMON_BSDF

/*
    Parameters for a BSDF implementation (assumes that the type
    of BSDF is already known, stored separately).
*/
struct BsdfData {
    uvec2 bits;
};

BsdfData create_bsdf_data(
    vec3 reflectance,
    float roughness)
{
    BsdfData bd;
    bd.bits.x = packHalf2x16(reflectance.xy);
    bd.bits.y = packHalf2x16(vec2(reflectance.z, roughness));
    return bd;
}

vec3 get_reflectance(BsdfData bd)
{
    return vec3(
        unpackHalf2x16(bd.bits.x),
        unpackHalf2x16(bd.bits.y).x);
}
float get_roughness(BsdfData bd)
{
    return unpackHalf2x16(bd.bits.y).y;
}

BsdfData replace_roughness(BsdfData bd, float roughness)
{
    bd.bits.y &= 0x0000ffff;
    bd.bits.y |= packHalf2x16(vec2(0.f, roughness));
    return bd;
}

#endif
