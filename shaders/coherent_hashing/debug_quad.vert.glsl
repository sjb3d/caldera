#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 0, scalar) uniform DebugQuadUniforms {
    vec2 scale;
    vec2 offset;
} g_debug_quad;

out gl_PerVertex {
    vec4 gl_Position;
};
layout(location = 0) out vec2 v_uv;

void main()
{
    vec2 uv = vec2(
        (gl_VertexIndex & 1) != 0 ? 1.f : 0.f,
        (gl_VertexIndex & 2) != 0 ? 0.f : 1.f);

    gl_Position = vec4(uv*g_debug_quad.scale + g_debug_quad.offset, 0.1, 1.f);
    v_uv = uv;
}
