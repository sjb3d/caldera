#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "color_space.glsl"
#include "sampler.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, scalar) uniform TraceData {
    uvec2 dims;
    vec2 dims_rcp;
    uint pass_index;
} g_trace;

layout(set = 0, binding = 1, r32f) uniform restrict image2D g_result_r;
layout(set = 0, binding = 2, r32f) uniform restrict image2D g_result_g;
layout(set = 0, binding = 3, r32f) uniform restrict image2D g_result_b;

layout(set = 0, binding = 4, rg16ui) uniform restrict readonly uimage2D g_samples;

#define SEQUENCE_COUNT        4096

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
    /*
        |o + t*d|^2 = r^2
        => (d.d)t^2 + 2(d.o)t + o.o - r^2 = 0
    */
    const vec3 o = ro - sc;

    const float a = dot(rd, rd);
    const float b = 2.f*dot(rd, o);
    const float c = dot(o, o) - sr*sr;

    const float q = b*b - 4.f*a*c;
    bool is_hit = false;
    if (q > 0.f) {
        const float s = .5f/a;
        const float ta = -b*s;
        const float tb = sqrt(q)*s;

        const float t0 = ta - tb;
        const float t1 = ta + tb;

        if (0.f < t0 && t0 < d) {
            gnv = o + t0*rd;
            d = t0;
            is_hit = true;
        }
        if (0.f < t1 && t1 < d) {
            gnv = o + t1*rd;
            d = t1;
            is_hit = true;
        }
    }
    return is_hit;
}

vec2 rand_u01(uvec2 pixel_coord, uint ray_index, uint sample_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((ray_index << 20) | (pixel_coord.y << 10) | pixel_coord.x);
    const ivec2 sample_coord = ivec2(sample_index, seq_hash & (SEQUENCE_COUNT - 1));
    const uvec2 sample_bits = imageLoad(g_samples, sample_coord).xy;
    return (vec2(sample_bits) + .5f)/65536.f;
}

float smith_lambda(vec3 v, vec2 alpha)
{
    return .5f*(sqrt(1.f + (alpha.x*alpha.x*v.x*v.x + alpha.y*alpha.y*v.y*v.y)/(v.z*v.z)) - 1.f);
}

float smith_g1(vec3 v, vec2 alpha)
{
    return 1.f/(1.f + smith_lambda(v, alpha));
}

float smith_g2(vec3 v, vec3 l, vec2 alpha)
{
    return 1.f/(1.f + smith_lambda(v, alpha) + smith_lambda(l, alpha));
}

vec3 schlick_fresnel(vec3 r0, float v_dot_h)
{
    return r0 + (vec3(1.f) - r0)*pow(1.f - v_dot_h, 5.f);
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
                float light_scale = 1.5f;
                vec3 light_value;
                if (((hit_bits.y >> 20) & 0x3) == ((hit_bits.x >> 21) & 0x3)) {
                    light_value = acescg_from_rec709(light_scale*vec3(1.f, 1.f, .5f));
                } else if (((hit_bits.y >> 20) & 0x3) == ((hit_bits.x >> 21) & 0x1)) {
                    light_value = acescg_from_rec709(light_scale*vec3(1.f, .1f, .5f));
                } else {
                    light_value = acescg_from_rec709(light_scale*vec3(.2f, .1f, .2f));
                }
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
            r0 = acescg_from_rec709(r0);

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
            const vec3 h = sample_ggx_vndf(out_dir, alpha, ray_rand_u01);
            const vec3 in_dir = reflect(-out_dir, h);

            // compute weight
            const vec3 weight = schlick_fresnel(r0, abs(dot(out_dir, h)))*smith_g2(out_dir, in_dir, alpha)/smith_g1(out_dir, alpha);
            sample_value *= weight;

            // continue ray
            ray_origin = hit_pos + eps_hack*normal;
            ray_dir = normalize(tangent*in_dir.x + bitangent*in_dir.y + normal*in_dir.z);
        }
    }
    vec3 col = sum/float(sample_count);
    if (g_trace.pass_index > 0) {
        col.x += imageLoad(g_result_r, ivec2(pixel_coord)).x;
        col.y += imageLoad(g_result_g, ivec2(pixel_coord)).x;
        col.z += imageLoad(g_result_b, ivec2(pixel_coord)).x;
    }
    imageStore(g_result_r, ivec2(pixel_coord), vec4(col.x));
    imageStore(g_result_g, ivec2(pixel_coord), vec4(col.y));
    imageStore(g_result_b, ivec2(pixel_coord), vec4(col.z));
}
