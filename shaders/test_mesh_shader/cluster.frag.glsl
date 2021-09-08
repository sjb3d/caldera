#version 460 core

layout(location = 0) in vec3 v_color;

layout(location = 0) out vec4 o_col;

void main()
{
    o_col = vec4(v_color, 0.f);
}
