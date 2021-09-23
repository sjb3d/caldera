const uint MAX_AGE = 15;

layout(set = 0, binding = 0, scalar) uniform HashTableUniforms {
    uint entry_count;
    uint offsets[MAX_AGE];
} g_table;

uint dilate2(uint x)
{
    // ref: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}
uint morton2d(uvec2 v)
{
    return (dilate2(v.y) << 1) | dilate2(v.x);
}

struct Entry {
    uint bits;
};

Entry make_entry(uint age, uint key, uint data)
{
    Entry e;
    e.bits = (age << 28) | (key << 8) | data;
    return e;
}
Entry make_entry(uint bits)
{
    Entry e;
    e.bits = bits;
    return e;
}
Entry increment_age(Entry e)
{
    e.bits += 0x10000000U;
    return e;
}
uint get_age(Entry e)   { return (e.bits & 0xf0000000U) >> 28; }
uint get_key(Entry e)   { return (e.bits & 0x0fffff00U) >> 8; }
uint get_data(Entry e)  { return (e.bits & 0x000000ffU); }

uint coherent_hash(uint key, uint age)
{
    return (key + g_table.offsets[age - 1]) % g_table.entry_count;
}
