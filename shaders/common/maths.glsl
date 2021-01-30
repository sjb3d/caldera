#ifndef INCLUDED_COMMON_MATHS
#define INCLUDED_COMMON_MATHS

float max_element(vec2 v)   { return max(v.x, v.y); }
float max_element(vec3 v)   { return max(max(v.x, v.y), v.z); }

#endif
