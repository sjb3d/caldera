#include "light_common.glsl"

void dome_light_eval(
    DomeLightParams params,
    vec3 target_position,
    vec3 light_extdir,
    out vec3 emission,
    out float solid_angle_pdf)
{
    emission = params.emission;
    solid_angle_pdf = 1.f/(4.f*PI);
}
