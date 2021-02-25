
#define SMITS_WAVELENGTH_MIN    380.f
#define SMITS_WAVELENGTH_MAX    720.f

float smits_power_from_rec709(float wavelength, vec3 col, sampler2D table)
{
    const float wavelength_u = unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelength);

    const vec4 rgbw = texture(table, vec2(.25f, wavelength_u));
    const vec3 cmy = texture(table, vec2(.75f, wavelength_u)).xyz;

    const float red = rgbw.x;
    const float gre = rgbw.y;
    const float blu = rgbw.z;
    const float whi = rgbw.w;
    const float cya = cmy.x;
    const float mag = cmy.y;
    const float yel = cmy.z;

    const float col_min = min_element(col);
    float ret = col_min*whi;
    if (col.x == col_min) {
        if (col.y < col.z) {
            ret += (col.y - col.x)*cya;
            ret += (col.z - col.y)*blu;
        } else {
            ret += (col.z - col.x)*cya;
            ret += (col.y - col.z)*gre;
        }
    } else if (col.y == col_min) {
        if (col.z < col.x) {
            ret += (col.z - col.y)*mag;
            ret += (col.x - col.z)*red;
        } else {
            ret += (col.x - col.y)*mag;
            ret += (col.z - col.x)*blu;
        }
    } else {
        if (col.x < col.y) {
            ret += (col.x - col.z)*yel;
            ret += (col.y - col.x)*gre;
        } else {
            ret += (col.y - col.z)*yel;
            ret += (col.x - col.y)*red;
        }
    }
    return ret;
}
