#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "smooth_dielectric_bsdf.glsl"

BSDF_SAMPLE_MAIN(smooth_dielectric_bsdf_sample);
