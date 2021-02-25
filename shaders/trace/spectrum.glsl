
#define SMITS_WAVELENGTH_MIN    380.f
#define SMITS_WAVELENGTH_MAX    720.f

float smits_power_from_rec709(float wavelength, vec3 col, sampler2D table)
{
    const float wavelength_u = (wavelength - SMITS_WAVELENGTH_MIN)/(SMITS_WAVELENGTH_MAX - SMITS_WAVELENGTH_MIN);

    const vec4 rgbw = texture(table, vec2(.25f, wavelength_u));
    const vec3 cmy = texture(table, vec2(.75f, wavelength_u)).xyz;

    const float col_min = min_element(col);
    float ret = col_min*rgbw.w; // white
    if (col.x == col_min) {
        if (col.y < col.z) {
            ret += (col.y - col.x)*cmy.x;   // cyan
            ret += (col.z - col.y)*rgbw.z;  // blue
        } else {
            ret += (col.z - col.x)*cmy.x;   // cyan
            ret += (col.y - col.z)*rgbw.y;  // green
        }
    } else if (col.y == col_min) {
        if (col.z < col.x) {
            ret += (col.z - col.y)*cmy.y;   // magenta
            ret += (col.x - col.z)*rgbw.x;  // red
        } else {
            ret += (col.x - col.y)*cmy.y;   // magenta
            ret += (col.z - col.x)*rgbw.z;  // blue
        }
    } else {
        if (col.x < col.y) {
            ret += (col.x - col.z)*cmy.z;   // yellow
            ret += (col.y - col.x)*rgbw.y;  // green
        } else {
            ret += (col.y - col.z)*cmy.z;   // yellow
            ret += (col.x - col.y)*rgbw.x;  // red
        }
    }
    return ret;
}
