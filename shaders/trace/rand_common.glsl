#include "sampler.glsl"

#define LOG2_SEQUENCE_COUNT     10

ivec2 rand_sample_coord(uvec2 pixel_coord, uint seq_index, uint sample_index)
{
    // hash the pixel coordinate and ray index to pick a sequence
    const uint seq_hash = hash((seq_index << 24) ^ (pixel_coord.y << 12) ^ pixel_coord.x);
    return ivec2(sample_index, seq_hash >> (32 - LOG2_SEQUENCE_COUNT));
}
