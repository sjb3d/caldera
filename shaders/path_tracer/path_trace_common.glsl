#include "light_common.glsl"

#define PATH_TRACE_FLAG_ACCUMULATE_ROUGHNESS                0x01
#define PATH_TRACE_FLAG_ALLOW_LIGHT_SAMPLING                0x02
#define PATH_TRACE_FLAG_ALLOW_BSDF_SAMPLING                 0x04
#define PATH_TRACE_FLAG_SPHERE_LIGHTS_SAMPLE_SOLID_ANGLE    0x08
#define PATH_TRACE_FLAG_PLANAR_LIGHTS_ARE_TWO_SIDED         0x10

#define MIS_HEURISTIC_NONE      0
#define MIS_HEURISTIC_BALANCE   1
#define MIS_HEURISTIC_POWER2    2

#define WAVELENGTH_SAMPLING_METHOD_UNIFORM          0
#define WAVELENGTH_SAMPLING_METHOD_HERO_MIS         1
#define WAVELENGTH_SAMPLING_METHOD_CONTINUOUS_MIS   2

struct CameraParams {
    mat4x3 world_from_local;
    vec2 fov_size_at_unit_z;
    float aperture_radius_ls;
    float focus_distance_ls;
    float pixel_size_at_unit_z;
};

layout(set = 0, binding = 0, scalar) uniform PathTraceUniforms {
    LightInfoTable light_info_table;
    LightAliasTable light_alias_table;
    uint64_t light_params_base;
    uint sampled_light_count;
    uint external_light_begin;
    uint external_light_end;
    CameraParams camera;
    uint sample_index;
    uint max_segment_count;
    uint mis_heuristic;
    uint sequence_type;
    uint wavelength_sampling_method;
    uint flags;
} g_path_trace;

layout(set = 0, binding = 1) uniform accelerationStructureEXT g_accel;
layout(set = 0, binding = 2, rg32f) uniform restrict readonly image2D g_pmj_samples;
layout(set = 0, binding = 3, rgba32ui) uniform restrict readonly uimage2D g_sobol_samples;
layout(set = 0, binding = 4) uniform sampler1DArray g_illuminants;
layout(set = 0, binding = 5) uniform sampler1DArray g_conductors;
layout(set = 0, binding = 6) uniform sampler2D g_smits_table;
layout(set = 0, binding = 7) uniform sampler1D g_wavelength_inv_cdf;
layout(set = 0, binding = 8) uniform sampler1D g_wavelength_pdf;
layout(set = 0, binding = 9) uniform sampler1D g_xyz_matching;
layout(set = 0, binding = 10, r32f) uniform restrict writeonly image2D g_result[3];
layout(set = 0, binding = 11) uniform sampler g_linear_sampler;

#define BINDLESS_MAX_SAMPLED_IMAGE_2D       1024

layout(set = 1, binding = 0) uniform texture2D g_textures[BINDLESS_MAX_SAMPLED_IMAGE_2D];
