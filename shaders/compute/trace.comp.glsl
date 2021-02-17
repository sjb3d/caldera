#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "tone_map.glsl"
#include "sampler.glsl"
#include "ggx.glsl"
#include "fresnel.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, scalar) uniform TraceData {
    uvec2 dims;
    vec2 dims_rcp;
    uint pass_index;
    uint render_color_space;
} g_trace;

layout(set = 0, binding = 1, r32f) uniform restrict image2D g_result[3];
layout(set = 0, binding = 2, rg16ui) uniform restrict readonly uimage2D g_samples;

#define LOG2_SEQUENCE_COUNT         12

void sample_camera(
    const vec2 film_uv,
    out vec3 ray_origin,
    out vec3 ray_dir)
{
    const vec3 camera_pos = vec3(-2.5f, .5f, 4.f);
    const vec3 look_at = vec3(0.f, -1.f, 0.f);

    const vec3 look_dir = normalize(look_at - camera_pos);
    const vec3 right_dir = normalize(cross(look_dir, vec3(0.f, 1.f, 0.f)));
    const vec3 up_dir = normalize(cross(right_dir, look_dir));

    const float tan_v_fov = tan(PI/6.f);
    const float aspect_ratio = g_trace.dims_rcp.y/g_trace.dims_rcp.x;

    const vec2 v = vec2(2.f*film_uv.x - 1.f, 1.f - 2.f*film_uv.y)*vec2(aspect_ratio*tan_v_fov, tan_v_fov);

    ray_origin = camera_pos;
    ray_dir = normalize(look_dir + v.x*right_dir + v.y*up_dir);
}

bool intersect_sphere(
    const vec3 ro,      // ray origin
    const vec3 rd,      // ray dir
    const vec3 sc,      // sphere centre
    const float sr,     // sphere radius
    inout vec3 gnv,     // geometric normal vector (unnormalised)
    inout float d)      // ray distance
{
    const vec3 o = ro - sc;
    vec2 t;
    const bool is_hit = ray_vs_sphere(o, rd, sr, t);

    bool is_closest_hit = false;
    if (is_hit) {
        const float t_min = min_element(t);
        const float t_max = max_element(t);
        if (0.f < t_min && t_min < d) {
            gnv = o + t_min*rd;
            d = t_min;
            is_closest_hit = true;
        }
        if (0.f < t_max && t_max < d) {
            gnv = o + t_max*rd;
            d = t_max;
            is_closest_hit = true;
        }
    }
    return is_closest_hit;
}

vec2 rand_u01(uvec2 pixel_coord, uint seq_index, uint sample_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((seq_index << 24) ^ (pixel_coord.y << 12) ^ pixel_coord.x);
    const ivec2 sample_coord = ivec2(sample_index, seq_hash >> (32 - LOG2_SEQUENCE_COUNT));
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

#define HIT_NONE    0
#define HIT_FLOOR   1
#define HIT_WALL    2
#define HIT_LIGHT   3
#define HIT_FOCUS   4

uint intersect_scene(vec3 ro, vec3 rd, inout vec3 gnv, inout float d)
{
    const float room_half_width = 3.f;
    const float room_half_height = 2.f;
    const float flat_rad = 10000.f;

    uint hit = HIT_NONE;
    if (intersect_sphere(ro, rd, vec3(0.f, -(room_half_height + flat_rad), 0.f), flat_rad, gnv, d)) {
        hit = HIT_FLOOR;
    }
    if (intersect_sphere(ro, rd, vec3(room_half_width + flat_rad, 0.f, 0.f), flat_rad, gnv, d)) {
        hit = HIT_WALL;
    }
    if (intersect_sphere(ro, rd, vec3(-(room_half_width + flat_rad), 0.f, 0.f), flat_rad, gnv, d)) {
        hit = HIT_WALL;
    }
    if (intersect_sphere(ro, rd, vec3(0.f, room_half_height + flat_rad, 0.f), flat_rad, gnv, d)) {
        hit = HIT_WALL;
    }
    if (intersect_sphere(ro, rd, vec3(0.f, 0.f, -(room_half_width + flat_rad)), flat_rad, gnv, d)) {
        hit = HIT_LIGHT;
    }
    if (intersect_sphere(ro, rd, vec3(1.f, 1.f - room_half_height, 1.f), 1.f, gnv, d)) {
        hit = HIT_FOCUS;
    }
    return hit;
}

void main()
{
    const uvec2 pixel_coord = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_coord, g_trace.dims))) {
        return;
    }

    float eps_hack = .01f;

    vec3 sum = vec3(0.f);
    const uint sample_count = 4;
    const uint max_ray_count = 4;
    for (uint sample_index = 0; sample_index < sample_count; ++sample_index) {
        const uint seq_sample_index = sample_count*g_trace.pass_index + sample_index;
        const vec2 film_rand_u01 = rand_u01(pixel_coord, 0, seq_sample_index);
        const vec2 film_uv = (vec2(pixel_coord) + film_rand_u01)*g_trace.dims_rcp;

        vec3 ray_origin, ray_dir;
        sample_camera(film_uv, ray_origin, ray_dir);

        vec3 sample_value = vec3(1.f);
        for (uint ray_index = 0; ray_index < max_ray_count; ++ray_index) {
            float d = FLT_INF;
            vec3 gnv = vec3(0.f, 0.f, 1.f);
            const uint hit = intersect_scene(ray_origin, ray_dir, gnv, d);
            if (hit == HIT_NONE) {
                break;
            }
            const vec3 hit_pos = ray_origin + ray_dir*d;

            if (hit == HIT_LIGHT) {
                const uvec2 hit_bits = floatBitsToUint(hit_pos.xy - vec2(4.f, 3.f));
                float light_scale = 1.0f;
                vec3 light_value;
                if (((hit_bits.y >> 20) & 0x3) == ((hit_bits.x >> 21) & 0x3)) {
                    light_value = light_scale*vec3(1.f, 1.f, .5f);
                } else if (((hit_bits.y >> 20) & 0x3) == ((hit_bits.x >> 21) & 0x1)) {
                    light_value = light_scale*vec3(1.f, .1f, .5f);
                } else {
                    light_value = light_scale*vec3(.2f, .1f, .2f);
                }
                light_value = sample_from_rec709(light_value, g_trace.render_color_space);
                sum += sample_value*light_value;
                break;
            }

            // adjust geometric normal to our hemisphere
            if (dot(gnv, ray_dir) > 0.f) {
                gnv = -gnv;
            }

            // get material
            vec3 r0 = vec3(.9f);
            vec2 alpha = vec2(.2f);
            if (hit == HIT_FOCUS) {
                r0 *= vec3(1.f, 1.f, .2f);
                alpha = vec2(.05f);
            } else if (hit == HIT_WALL) {
                r0 *= vec3(.5f, .5f, 1.f);
            } else if (hit == HIT_FLOOR) {
                if ((fract(hit_pos.x) < .5f) == ((fract(hit_pos.z) < .5f))) {
                    alpha *= 2.f;
                }
            }
            r0 = sample_from_rec709(r0, g_trace.render_color_space);

            // make sampling basis
            const vec3 normal = normalize(gnv);
            const vec3 tangent = normalize(perp(gnv));
            const vec3 bitangent = cross(normal, tangent);

            // get outgoing ray in this basis
            const vec3 out_dir = vec3(
                -dot(ray_dir, tangent),
                -dot(ray_dir, bitangent),
                -dot(ray_dir, normal));

            // sample GGX
            const vec2 ray_rand_u01 = rand_u01(pixel_coord, 1 + ray_index, seq_sample_index);
            const vec3 h = sample_vndf(out_dir, alpha, ray_rand_u01);
            const vec3 in_dir = reflect(-out_dir, h);

            // compute estimator
            const vec3 estimator
                = fresnel_schlick(r0, dot(out_dir, h))
                * ggx_vndf_sampled_estimator_without_fresnel(out_dir, in_dir, alpha);
            sample_value *= estimator;

            // continue ray
            ray_origin = hit_pos + eps_hack*normal;
            ray_dir = normalize(tangent*in_dir.x + bitangent*in_dir.y + normal*in_dir.z);
        }
    }
    vec3 col = sum/float(sample_count);
    if (g_trace.pass_index > 0) {
        col.x += imageLoad(g_result[0], ivec2(pixel_coord)).x;
        col.y += imageLoad(g_result[1], ivec2(pixel_coord)).x;
        col.z += imageLoad(g_result[2], ivec2(pixel_coord)).x;
    }
    imageStore(g_result[0], ivec2(pixel_coord), vec4(col.x));
    imageStore(g_result[1], ivec2(pixel_coord), vec4(col.y));
    imageStore(g_result[2], ivec2(pixel_coord), vec4(col.z));
}
