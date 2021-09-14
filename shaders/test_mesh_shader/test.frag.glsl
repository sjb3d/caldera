#version 460 core

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal_viewspace;
layout(location = 2) in vec3 v_pos_viewspace;

layout(location = 0) out vec4 o_col;

void main()
{
    vec3 N = normalize(v_normal_viewspace);
    vec3 V = normalize(-v_pos_viewspace);
    vec3 L = vec3(0.f, 0.f, 1.f);
    float N_dot_L = dot(N, L);

    vec3 col = mix(0.2f, 1.f, max(N_dot_L, 0.f)) * v_color;
    o_col = vec4(col, 1.f);
}
