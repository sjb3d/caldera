#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "tone_map.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, scalar) uniform CaptureData {
    uvec2 size;
    float exposure_scale;
    mat3 rec709_from_xyz;
    mat3 acescg_from_xyz;
    uint tone_map_method;
} g_capture;

layout(set = 0, binding = 1, scalar) restrict writeonly buffer CaptureOutput {
    uint arr[];
} g_output;

layout(set = 0, binding = 2, rgba32f) uniform restrict readonly image2D g_input;

void main()
{
    const uvec2 pixel_coord = uvec2(4*gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    if (any(greaterThanEqual(pixel_coord, g_capture.size))) {
        return;
    }

    const vec4 p0 = imageLoad(g_input, ivec2(pixel_coord.x + 0, pixel_coord.y));
    const vec4 p1 = imageLoad(g_input, ivec2(pixel_coord.x + 1, pixel_coord.y));
    const vec4 p2 = imageLoad(g_input, ivec2(pixel_coord.x + 2, pixel_coord.y));
    const vec4 p3 = imageLoad(g_input, ivec2(pixel_coord.x + 3, pixel_coord.y));

    const vec3 sample0 = p0.xyz*(g_capture.exposure_scale/p0.w);
    const vec3 sample1 = p1.xyz*(g_capture.exposure_scale/p1.w);
    const vec3 sample2 = p2.xyz*(g_capture.exposure_scale/p2.w);
    const vec3 sample3 = p3.xyz*(g_capture.exposure_scale/p3.w);

    const vec3 col0 = tone_map_sample_to_gamma(sample0, g_capture.rec709_from_xyz, g_capture.acescg_from_xyz, g_capture.tone_map_method);
    const vec3 col1 = tone_map_sample_to_gamma(sample1, g_capture.rec709_from_xyz, g_capture.acescg_from_xyz, g_capture.tone_map_method);
    const vec3 col2 = tone_map_sample_to_gamma(sample2, g_capture.rec709_from_xyz, g_capture.acescg_from_xyz, g_capture.tone_map_method);
    const vec3 col3 = tone_map_sample_to_gamma(sample3, g_capture.rec709_from_xyz, g_capture.acescg_from_xyz, g_capture.tone_map_method);

    const uint w0 = packUnorm4x8(vec4(col0.xyz, col1.x));
    const uint w1 = packUnorm4x8(vec4(col1.yz, col2.xy));
    const uint w2 = packUnorm4x8(vec4(col2.z, col3.xyz));

    const uint base_offset = (((pixel_coord.y*g_capture.size.x) + pixel_coord.x)*3)/4;
    g_output.arr[base_offset + 0] = w0;
    g_output.arr[base_offset + 1] = w1;
    g_output.arr[base_offset + 2] = w2;
}
