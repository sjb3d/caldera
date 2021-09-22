#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) out vec4 o_col;

layout(set = 0, binding = 0, r8) uniform readonly image2D g_image;

void main()
{
    uvec2 coord = uvec2(gl_FragCoord.xy);
    vec3 col = vec3(.5f);
    if (all(lessThan(coord, uvec2(1024)))) {
        vec2 uv = (vec2(coord) + vec2(.5f))/1024.f;
        float s = imageLoad(g_image, ivec2(coord)).x;
        col = mix(vec3(1.f), vec3(uv, 0.f), s);
    }
    o_col = vec4(col, 1.f);
}
