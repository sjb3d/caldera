#version 430 core

out gl_PerVertex {
    vec4 gl_Position;
};

void main()
{
    vec2 v = vec2(
        gl_VertexIndex == 1 ? 3.f : -1.f,
        gl_VertexIndex == 2 ? 3.f : -1.f);
    gl_Position = vec4(v, 0.1, 1.f);
}
