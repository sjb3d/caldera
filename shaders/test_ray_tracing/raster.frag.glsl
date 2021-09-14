#version 460 core

layout(location = 0) in vec3 v_normal;

layout(location = 0) out vec4 o_col;

void main()
{
    const vec3 normal = normalize(v_normal);

    o_col = vec4(.5f*normal + .5f, 0.f);
}
