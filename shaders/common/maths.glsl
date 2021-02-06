#ifndef INCLUDED_COMMON_MATHS
#define INCLUDED_COMMON_MATHS

#define PI          3.1415926535f

#define FLT_INF     uintBitsToFloat(0x7f800000) 

float sum_elements(vec2 v)  { return v.x + v.y; }
float sum_elements(vec3 v)  { return v.x + v.y + v.z; }

float mul_elements(vec2 v)  { return v.x * v.y; }
float mul_elements(vec3 v)  { return v.x * v.y * v.z; }

float max_element(vec2 v)   { return max(v.x, v.y); }
float max_element(vec3 v)   { return max(max(v.x, v.y), v.z); }

float copysign(float x, float s)
{
    const uint x_bits = floatBitsToUint(x);
    const uint s_bits = floatBitsToUint(s);
    const uint y_bits = (x_bits & 0x7fffffff) | (s_bits & 0x80000000);
    return uintBitsToFloat(y_bits);
}
vec2 copysign(vec2 x, vec2 s)
{
    const uvec2 x_bits = floatBitsToUint(x);
    const uvec2 s_bits = floatBitsToUint(s);
    const uvec2 y_bits = (x_bits & 0x7fffffff) | (s_bits & 0x80000000);
    return uintBitsToFloat(y_bits);
}

// reference: https://blog.selfshadow.com/2011/10/17/perp-vectors/
vec3 perp(vec3 u)
{
    const vec3 a = abs(u);

    const uint uyx = floatBitsToUint(a.x - a.y) >> 31;
    const uint uzx = floatBitsToUint(a.x - a.z) >> 31;
    const uint uzy = floatBitsToUint(a.y - a.z) >> 31;

    const uint xm = uyx & uzx;
    const uint ym = (1 ^ xm) & uzy;
    const uint zm = 1 ^ (xm & ym);

    const float xf = (xm != 0) ? 1.f : 0.f;
    const float yf = (ym != 0) ? 1.f : 0.f;
    const float zf = (zm != 0) ? 1.f : 0.f;

    return cross(u, vec3(xf, yf, zf));
}

mat3 basis_from_z_axis(vec3 z_axis)
{
    const vec3 x_axis = normalize(perp(z_axis));
    const vec3 y_axis = normalize(cross(z_axis, x_axis));
    return mat3(x_axis, y_axis, z_axis);
}

#endif
