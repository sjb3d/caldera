#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#extension GL_GOOGLE_include_directive : require
#include "diffuse_bsdf.glsl"

BSDF_SAMPLE_MAIN(diffuse_bsdf_sample);
