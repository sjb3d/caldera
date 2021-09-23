#version 460 core
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "hash_table_common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 1, scalar) restrict writeonly buffer Entries {
    uint arr[];
} g_entries;

layout(set = 0, binding = 2, r8) uniform restrict readonly image2D g_image;

bool insert_entry(uint key, uint data)
{
   Entry entry = make_entry(1, key, data);
    for (;;) {
        uint entry_index = coherent_hash(get_key(entry), get_age(entry));
        Entry prev = make_entry(atomicMax(g_entries.arr[entry_index], entry.bits));
        if (entry.bits > prev.bits) {
            // TODO: update max age for (key, 1)
            if (get_age(prev) == 0) {
                // slot was unused, we are done
                return true;
            } else {
                // continue with the entry we just replaced
                entry = prev;
            }
        }
        if (get_age(entry) == MAX_AGE) {
            // insert failed
            return false;
        }
        entry = increment_age(entry);       
    }
}

void main()
{
    uvec2 coord = gl_GlobalInvocationID.xy;

    uint key = morton2d(coord);
    uint data = uint(clamp(int(imageLoad(g_image, ivec2(coord)).x*255.f + .5f), 0, 255));
    insert_entry(key, data);
}
