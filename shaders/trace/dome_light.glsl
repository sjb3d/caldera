#include "light_common.glsl"

void dome_light_eval(
    DomeLightParams params,
    vec3 target_position,
    vec3 light_extdir,
    out vec3 illuminant_tint,
    out float solid_angle_pdf)
{
    illuminant_tint = params.illuminant_tint;
    solid_angle_pdf = 1.f/(4.f*PI);
}
