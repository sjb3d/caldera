// reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
// reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// see color_space::derive_matrices() for derivation
vec3 acescg_from_rec709(vec3 col)
{
    // ap1_from_xyz * d60_from_d65 * xyz_from_rec709
    return col.x*vec3(0.6131948f, 0.07020554f, 0.020618888f)
        +  col.y*vec3(0.33951503f, 0.91633457f, 0.10956727f)
        +  col.z*vec3(0.047367923f, 0.013449377f, 0.86960644f);
}

// reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
// reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// see color_space::derive_matrices() for derivation
vec3 rec709_from_acescg(vec3 col)
{
    // rec709_from_xyz * d65_from_d60 * xyz_from_ap1
    return col.x*vec3(1.7047806f, -0.13026042f, -0.02400902)
        +  col.y*vec3(-0.62169176f, 1.1408291f, -0.12899965f)
        +  col.z*vec3(-0.08324518f, -0.010548766f, 1.1532483f);
}

// reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/rrt/RRT.ctl (RRT_SAT_MAT)
// input is AP1, output is desaturated input for rrt_and_odt_fit
vec3 rrt_sat(vec3 col)
{
    const vec3 y_from_ap1 = vec3(0.2722287168f, 0.6740817658f, 0.0536895174f);
    const float rrt_sat_factor = 0.96f;
    return mix(dot(col, y_from_ap1).xxx, col, rrt_sat_factor);
}

// reference: https://github.com/ampas/aces-dev/blob/master/transforms/ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl (ODT_SAT_MAT)
// input is the result of rrt_and_odt_fit before desaturation, output is AP1
vec3 odt_sat(vec3 col)
{
    const vec3 y_from_ap1 = vec3(0.2722287168f, 0.6740817658f, 0.0536895174f);
    const float odt_sat_factor = 0.93f;
    return mix(dot(col, y_from_ap1).xxx, col, odt_sat_factor);
}

// reference: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// fits the spline parts of RRT and ODT
vec3 odt_and_rrt_fit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 fit_from_rec709(vec3 col)
{
    // equivalent to rrt_sat(acescg_from_rec709(col))
    return col.x*vec3(0.59719f, 0.07600f, 0.02840f)
        +  col.y*vec3(0.35458f, 0.90834f, 0.13383f)
        +  col.z*vec3(0.04823f, 0.01566f, 0.83777f);
}
vec3 rec709_from_fit(vec3 col)
{
    // equivalent to rec709_from_acescg(odt_sat(col))
    return col.x*vec3( 1.60475f, -0.10208f, -0.00327f)
        +  col.y*vec3(-0.53108f,  1.10813f, -0.07276f)
        +  col.z*vec3(-0.07367f, -0.00605f,  1.07602f);
}

float linear_from_gamma(float x)
{
    const float lo = x/12.92f;
    const float hi = pow((x + 0.055f)/1.055f, 2.4f);
    return (x < 0.04045f) ? lo : hi;
}
vec3 linear_from_gamma(vec3 c)
{
    return vec3(
        linear_from_gamma(c.x),
        linear_from_gamma(c.y),
        linear_from_gamma(c.z));
}

float gamma_from_linear(float x)
{
    const float lo = x*12.92f;
    const float hi = 1.055f*pow(x, 1.f/2.4f) - 0.055f;
    return (x < 0.0031308f) ? lo : hi;
}
vec3 gamma_from_linear(vec3 c)
{
    return vec3(
        gamma_from_linear(c.x),
        gamma_from_linear(c.y),
        gamma_from_linear(c.z));
}

// reference: http://filmicworlds.com/blog/filmic-tonemapping-operators/
float filmic_tone_map(float x)
{
    x = max(0.f, x - .004f);
    x = (x*(6.2f*x + .5f)) / (x*(6.2f*x + 1.7f) + .06f);
    return x;
}
vec3 filmic_tone_map(vec3 c)
{
    return vec3(
        filmic_tone_map(c.x),
        filmic_tone_map(c.y),
        filmic_tone_map(c.z));
}
