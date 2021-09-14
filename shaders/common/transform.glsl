#ifndef INCLUDED_COMMON_TRANSFORM
#define INCLUDED_COMMON_TRANSFORM

#include "maths.glsl"

vec3 quat_rotate(vec4 q, vec3 v)
{
    // ref: https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    vec3 t = cross(2.f*q.xyz, v);
    return v + q.w*t + cross(q.xyz, t);
}

struct PackedTransform
{
    vec3 translation;
    float scale;
    vec4 rotation_quat;
};

vec3 transform_point(PackedTransform t, vec3 p)
{
    return quat_rotate(t.rotation_quat, p)*t.scale + t.translation;
}
vec3 transform_vec(PackedTransform t, vec3 v)
{
    return quat_rotate(t.rotation_quat, v)*t.scale;
}
vec3 transform_unit(PackedTransform t, vec3 v)
{
    return flip_sign(quat_rotate(t.rotation_quat, v), vec3(t.scale));
}

#endif // ndef INCLUDED_COMMON_TRANSFORM
