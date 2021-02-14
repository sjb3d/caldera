
float ggx_d(vec3 h, vec2 alpha)
{
    const vec3 ha = vec3(h.xy/alpha, h.z);
    const float m = sum_elements(ha*ha);
    return 1.f/(PI*alpha.x*alpha.y*m*m);
}

float smith_lambda(vec3 v, vec2 alpha)
{
    const vec3 va = vec3(v.xy*alpha, v.z);
    const vec3 va2 = va*va;
    return .5f*(sqrt(1.f + (va2.x + va2.y)/va2.z) - 1.f);
}

float smith_g1(vec3 v, vec2 alpha)
{
    return 1.f/(1.f + smith_lambda(v, alpha));
}

float smith_g2(vec3 v, vec3 l, vec2 alpha)
{
    return 1.f/(1.f + smith_lambda(v, alpha) + smith_lambda(l, alpha));
}

vec3 schlick_fresnel(vec3 r0, float h_dot_v)
{
    h_dot_v = abs(h_dot_v);
    return r0 + (vec3(1.f) - r0)*pow(1.f - h_dot_v, 5.f);
}
float schlick_fresnel(float r0, float h_dot_v)
{
    h_dot_v = abs(h_dot_v);
    return r0 + (1.f - r0)*pow(1.f - h_dot_v, 5.f);
}

vec3 ggx_brdf(vec3 r0, vec3 h, float h_dot_v, vec3 v, vec3 l, vec2 alpha)
{
    const float n_dot_v = abs(v.z);
    const float n_dot_l = abs(l.z);
    return schlick_fresnel(r0, h_dot_v) * ggx_d(h, alpha) * smith_g2(v, l, alpha) / (4.f * n_dot_v * n_dot_l);
}
float ggx_brdf(float r0, vec3 h, float h_dot_v, vec3 v, vec3 l, vec2 alpha)
{
    const float n_dot_v = abs(v.z);
    const float n_dot_l = abs(l.z);
    return schlick_fresnel(r0, h_dot_v) * ggx_d(h, alpha) * smith_g2(v, l, alpha) / (4.f * n_dot_v * n_dot_l);
}

// reference: http://jcgt.org/published/0007/04/01/
vec3 sample_vndf(vec3 Ve, vec2 alpha, vec2 u)
{
    // transforming the view direction to the hemisphere configuration
    const vec3 Vh = normalize(vec3(alpha.x*Ve.x, alpha.y*Ve.y, Ve.z));
    // orthonormal basis
    const float lensq = dot(Vh.xy, Vh.xy);
    const vec3 T1 = (lensq > 0.f) ? (vec3(-Vh.y, Vh.x, 0.f)/sqrt(lensq)) : vec3(1.f, 0.f, 0.f);
    const vec3 T2 = cross(Vh, T1);
    // parameterization of the projected area
    vec2 p = sample_disc_uniform(u);
    const float s = .5f*(1.f + Vh.z);
    p.y = mix(sqrt(max(1.f - p.x*p.x, 0.f)), p.y, s);
    // reprojection onto hemisphere
    const vec3 Nh = p.x*T1 + p.y*T2 + sqrt(max(0.f, 1.f - dot(p, p)))*Vh;
    // transforming the normal back to the ellipsoid configuration
    return normalize(vec3(alpha.x*Nh.x, alpha.y*Nh.y, max(0.f, Nh.z)));
}

float vndf_pdf(vec3 v, vec3 h, float h_dot_v, vec2 alpha)
{
    const float n_dot_v = abs(v.z);
    return smith_g1(v, alpha) * h_dot_v * ggx_d(h, alpha) / n_dot_v;
}

float ggx_vndf_sampled_pdf(vec3 v, vec3 h, vec2 alpha)
{
    // Algebraic simplification of: vndf_pdf / (4.f * h_dot_v)
    const float n_dot_v = abs(v.z);
    return smith_g1(v, alpha) * ggx_d(h, alpha) / (4.f * n_dot_v);
}

// reference: http://jcgt.org/published/0007/04/01/
vec3 ggx_vndf_sampled_estimator(vec3 r0, float h_dot_v, vec3 v, vec3 l, vec2 alpha)
{
    // Algebraic simplification of: ggx_brdf * n_dot_l / ggx_vndf_sampled_pdf
    return schlick_fresnel(r0, h_dot_v) * smith_g2(v, l, alpha) / smith_g1(v, alpha);
}
float ggx_vndf_sampled_estimator(float r0, float h_dot_v, vec3 v, vec3 l, vec2 alpha)
{
    return schlick_fresnel(r0, h_dot_v) * smith_g2(v, l, alpha) / smith_g1(v, alpha);
}
