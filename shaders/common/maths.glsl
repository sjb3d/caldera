#ifndef INCLUDED_COMMON_MATHS
#define INCLUDED_COMMON_MATHS

#define PI          3.1415926535f

#define FLT_BITS_SIGN       0x80000000
#define FLT_BITS_NOTSIGN    0x7FFFFFFF

#define FLT_INF     uintBitsToFloat(0x7f800000) 

float sum_elements(vec2 v)  { return v.x + v.y; }
float sum_elements(vec3 v)  { return v.x + v.y + v.z; }

float mul_elements(vec2 v)  { return v.x * v.y; }
float mul_elements(vec3 v)  { return v.x * v.y * v.z; }

float min_element(vec2 v)   { return min(v.x, v.y); }
float min_element(vec3 v)   { return min(min(v.x, v.y), v.z); }

float max_element(vec2 v)   { return max(v.x, v.y); }
float max_element(vec3 v)   { return max(max(v.x, v.y), v.z); }

float square(float x)       { return x*x; }
vec2 square(vec2 x)         { return x*x; }
vec3 square(vec3 x)         { return x*x; }

float unlerp(float a, float b, float x)
{
    return (x - a)/(b - a);
}

float copysign(float x, float s)
{
    const uint x_bits = floatBitsToUint(x);
    const uint s_bits = floatBitsToUint(s);
    const uint y_bits = (x_bits & FLT_BITS_NOTSIGN) | (s_bits & FLT_BITS_SIGN);
    return uintBitsToFloat(y_bits);
}
vec2 copysign(vec2 x, vec2 s)
{
    const uvec2 x_bits = floatBitsToUint(x);
    const uvec2 s_bits = floatBitsToUint(s);
    const uvec2 y_bits = (x_bits & FLT_BITS_NOTSIGN) | (s_bits & FLT_BITS_SIGN);
    return uintBitsToFloat(y_bits);
}

bool sign_bit_set(float x)
{
    const uint x_bits = floatBitsToUint(x);
    return ((x_bits & FLT_BITS_SIGN) != 0);
}
bool sign_bits_match(float x, float y)
{
    const uint x_bits = floatBitsToUint(x);
    const uint y_bits = floatBitsToUint(y);
    return (((x_bits ^ y_bits) & FLT_BITS_SIGN) == 0);
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

bool ray_vs_sphere(vec3 p, vec3 d, float r, out vec2 t)
{
    const float a = dot(d, d);
    const float c = dot(p, p) - r*r;

    // compute the discriminant in this way to avoid precision loss
    // reference: Ray Tracing Gems, Chapter 7
    const float b = -dot(p, d);
    const vec3 tmp = p + d*(b/a);
    const float q = r*r - dot(tmp, tmp);
    
    const bool has_hit = (q > 0.f);
    if (has_hit) {
        const float k = b + copysign(sqrt(a*q), b);
        t = vec2(k/a, c/k);
    }
    return has_hit;
}

vec2 ray_vs_sphere_force_hit(vec3 p, vec3 d, float r)
{
    const float a = dot(d, d);
    const float c = dot(p, p) - r*r;

    // compute the discriminant in this way to avoid precision loss
    // reference: Ray Tracing Gems, Chapter 7
    const float b = -dot(p, d);
    const vec3 tmp = p + d*(b/a);
    const float q = max(r*r - dot(tmp, tmp), 0.f);
    
    const float k = b + copysign(sqrt(a*q), b);
    return vec2(k/a, c/k);
}

#endif
