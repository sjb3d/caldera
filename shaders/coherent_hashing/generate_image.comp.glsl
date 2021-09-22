#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, r8) uniform restrict writeonly image2D g_image; 

void main()
{
    uvec2 coord = gl_GlobalInvocationID.xy;

    vec2 p = (vec2(coord) - vec2(512.0));
    float d = abs(sqrt(dot(p, p)) - 256.0) - 2.5;
    float s = smoothstep(0.8, -0.8, d);

    imageStore(g_image, ivec2(coord), vec4(s));
}
