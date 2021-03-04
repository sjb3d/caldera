#include "light_common.glsl"

void dome_light_eval(
    uint64_t params_addr,
    vec3 target_position,
    vec3 light_extdir,
    out vec3 illuminant_tint,
    out float solid_angle_pdf)
{
    DomeLightParamsBuffer buf = DomeLightParamsBuffer(params_addr);

    illuminant_tint = buf.params.illuminant_tint;
    solid_angle_pdf = 1.f/(4.f*PI);
}
