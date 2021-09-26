#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 1, r8) uniform readonly image2D g_image;

layout(location = 0) out vec4 o_col;

layout(location = 0) in vec2 v_uv;

void main()
{
    uvec2 coord = uvec2(1024.f*v_uv + .5f);
    float s = imageLoad(g_image, ivec2(coord)).x;
    vec3 col = mix(vec3(1.f), vec3(v_uv, 0.f), s);
    o_col = vec4(col, 1.f);
}
