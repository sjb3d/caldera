#version 460 core
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 16, local_size_y = 16) in;

struct CircleParams {
    vec2 centre;
    float radius;
};
const uint CIRCLE_COUNT = 4;

layout(set = 0, binding = 0, scalar) uniform GenerateImageUniforms {
    CircleParams circles[CIRCLE_COUNT];
} g_generate_image;

layout(set = 0, binding = 1, r8) uniform restrict writeonly image2D g_image; 

void main()
{
    uvec2 coord = gl_GlobalInvocationID.xy;

    float s = 0.f;
    for (uint i = 0; i < CIRCLE_COUNT; ++i) {
        CircleParams params = g_generate_image.circles[i];
        vec2 p = (vec2(coord) - params.centre);
        float d = abs(sqrt(dot(p, p)) - params.radius) - 2.5;
        s = max(s, smoothstep(0.8, -0.8, d));
    }
    imageStore(g_image, ivec2(coord), vec4(s));
}
