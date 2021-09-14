
#define SMITS_WAVELENGTH_MIN    380.f
#define SMITS_WAVELENGTH_MAX    720.f

float offset_wavelength(float base, float fraction)
{
    float result = base + fraction*(SMITS_WAVELENGTH_MAX - SMITS_WAVELENGTH_MIN);
    if (result > SMITS_WAVELENGTH_MAX) {
        result -= (SMITS_WAVELENGTH_MAX - SMITS_WAVELENGTH_MIN);
    }
    return result;
}

#define WAVELENGTHS_PER_RAY     3
#define HERO_VEC                vec3

HERO_VEC expand_wavelengths_from_hero(float hero_wavelength)
{
    HERO_VEC wavelengths;
    wavelengths.x = hero_wavelength;
    wavelengths.y = offset_wavelength(hero_wavelength, 1.f/WAVELENGTHS_PER_RAY);
    wavelengths.z = offset_wavelength(hero_wavelength, 2.f/WAVELENGTHS_PER_RAY);
    return wavelengths;
}

float smits_power_from_rec709(float wavelength, vec3 col, sampler2D table)
{
    const float wavelength_u = unlerp(SMITS_WAVELENGTH_MIN, SMITS_WAVELENGTH_MAX, wavelength);

    const vec4 wcmy = texture(table, vec2(.25f, wavelength_u));
    const vec4 rgb_ = texture(table, vec2(.75f, wavelength_u));

    const float whi = wcmy.x;
    const float cya = wcmy.y;
    const float mag = wcmy.z;
    const float yel = wcmy.w;
    const float red = rgb_.x;
    const float gre = rgb_.y;
    const float blu = rgb_.z;

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
