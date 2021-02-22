// expects globals for g_pmj_samples/g_sobol_samples, since GLSL cannot pass images as parameters!

#define PMJ_SEQUENCE_COUNT      1024
#define SOBOL_SAMPLE_COUNT      1024

#define SEQUENCE_TYPE_PMJ       0
#define SEQUENCE_TYPE_SOBOL     1

vec2 rand_u01(uvec2 pixel_coord, uint seq_index, uint sample_index, uint sequence_type)
{
    const uint seq_hash = get_seq_hash(pixel_coord, seq_index, sample_index);
    vec2 u;
    switch (sequence_type) {
        case SEQUENCE_TYPE_PMJ: {
            u = imageLoad(g_pmj_samples, ivec2(sample_index, seq_hash % PMJ_SEQUENCE_COUNT)).xy;
        } break;

        case SEQUENCE_TYPE_SOBOL: {
            const uint index = nested_uniform_shuffle(sample_index, seq_hash) % SOBOL_SAMPLE_COUNT;
            const uvec2 sobol = imageLoad(g_sobol_samples, ivec2(index, 0)).xy;
            const uint scramble0 = xorshift32(seq_hash);
            const uint scramble1 = xorshift32(scramble0);
            u = vec2(
                unit_float_from_high_bits(nested_uniform_shuffle(sobol.x, scramble0)),
                unit_float_from_high_bits(nested_uniform_shuffle(sobol.y, scramble1)));
        } break;
    }
    return u;
}
