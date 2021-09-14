#version 460 core

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal_viewspace;
layout(location = 2) in vec3 v_pos_viewspace;

layout(location = 0) out vec4 o_col;

void main()
{
    vec3 N = normalize(v_normal_viewspace);
    vec3 V = normalize(-v_pos_viewspace);
    vec3 L = normalize(vec3(.5f, .5f, 1.f));
    vec3 H = normalize(V + L);
    float N_dot_L = dot(N, L);
    float N_dot_H = dot(N, H);

    vec3 col
        = mix(0.2f, 1.f, max(N_dot_L, 0.f))*v_color
        + 0.2f*pow(max(N_dot_H, 0.f), 64.f)*clamp(8.f*N_dot_L, 0.f, 1.f)
        ;
    o_col = vec4(col, 1.f);
}
