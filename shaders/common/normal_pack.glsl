
float sum_elements(vec2 v)
{
    return v.x + v.y;
}
float sum_elements(vec3 v)
{
    return v.x + v.y + v.z;
}

vec2 copysign(vec2 x, vec2 s)
{
    const uvec2 x_bits = floatBitsToUint(x);
    const uvec2 s_bits = floatBitsToUint(s);
    const uvec2 y_bits = (x_bits & 0x7fffffff) | (s_bits & 0x80000000);
    return uintBitsToFloat(y_bits);
}

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
