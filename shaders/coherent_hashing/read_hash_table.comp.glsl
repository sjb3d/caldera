#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 1, scalar) restrict readonly buffer Entries {
    uint arr[];
} g_entries;

layout(set = 0, binding = 2, r8) uniform restrict writeonly image2D g_image;

bool get_entry(uint key, out uint data)
{
    for (uint age = 1; age <= MAX_AGE; ++age) {
        uint entry_index = coherent_hash(key, age);
        Entry entry = make_entry(g_entries.arr[entry_index]);
        if (get_age(entry) == 0) {
            // entry is empty, no need to check older ones
            break;
        }
        if (get_key(entry) == key) {
            data = get_data(entry);
            return true;
        }
    }
    data = 0;
    return false;
}

void main()
{
    uvec2 coord = gl_GlobalInvocationID.xy;

    uint key = morton2d(coord);
    
    uint data;
    get_entry(key, data);

    imageStore(g_image, ivec2(coord), vec4(float(data)/255.f));
}
