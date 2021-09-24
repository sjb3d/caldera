GLSLC=glslangValidator
GLSLCFLAGS_COMMON=-V -Ishaders/common
GLSLCFLAGS_MESH=--target-env spirv1.3
GLSLCFLAGS_RAYS=--target-env spirv1.4

DISASM=spirv-dis

INC=\
	coherent_hashing/hash_table_common.glsl \
	common/color_space.glsl \
	common/fresnel.glsl \
	common/ggx.glsl \
	common/maths.glsl \
	common/normal_pack.glsl \
	common/sampler.glsl \
	common/tone_map.glsl \
	common/transform.glsl \
	path_tracer/bsdf_common.glsl \
	path_tracer/diffuse_bsdf.glsl \
	path_tracer/disc_light.glsl \
	path_tracer/dome_light.glsl \
	path_tracer/extend_common.glsl \
	path_tracer/light_common.glsl \
	path_tracer/mirror_bsdf.glsl \
	path_tracer/occlusion_common.glsl \
	path_tracer/path_trace_common.glsl \
	path_tracer/quad_light.glsl \
	path_tracer/rough_conductor_bsdf.glsl \
	path_tracer/rough_dielectric_bsdf.glsl \
	path_tracer/rough_plastic_bsdf.glsl \
	path_tracer/sequence.glsl \
	path_tracer/solid_angle_light.glsl \
	path_tracer/spectrum.glsl \
	path_tracer/sphere_light.glsl \
	path_tracer/smooth_dielectric_bsdf.glsl \
	path_tracer/smooth_plastic_bsdf.glsl \
	path_tracer/triangle_mesh_light.glsl \
	test_mesh_shader/cluster_common.glsl

SRC=\
	coherent_hashing/clear_hash_table.comp.glsl \
	coherent_hashing/debug_image.frag.glsl \
	coherent_hashing/debug_quad.vert.glsl \
	coherent_hashing/generate_image.comp.glsl \
	coherent_hashing/read_hash_table.comp.glsl \
	coherent_hashing/write_hash_table.comp.glsl \
	path_tracer/capture.comp.glsl \
	path_tracer/copy.frag.glsl \
	path_tracer/copy.vert.glsl \
	path_tracer/disc.rint.glsl \
	path_tracer/extend.rmiss.glsl \
	path_tracer/extend_procedural.rchit.glsl \
	path_tracer/extend_triangle.rchit.glsl \
	path_tracer/filter.comp.glsl \
	path_tracer/mandelbulb.rint.glsl \
	path_tracer/occlusion.rchit.glsl \
	path_tracer/occlusion.rmiss.glsl \
	path_tracer/path_trace.rgen.glsl \
	path_tracer/sphere.rint.glsl \
	test_compute/trace.comp.glsl \
	test_compute/copy.frag.glsl \
	test_compute/copy.vert.glsl \
	test_mesh_shader/cluster.mesh.glsl \
	test_mesh_shader/cluster.task.glsl \
	test_mesh_shader/standard.vert.glsl \
	test_mesh_shader/test.frag.glsl \
	test_ray_tracing/copy.frag.glsl \
	test_ray_tracing/copy.vert.glsl \
	test_ray_tracing/raster.vert.glsl \
	test_ray_tracing/raster.frag.glsl \
	test_ray_tracing/trace.rchit.glsl \
	test_ray_tracing/trace.rgen.glsl \
	test_ray_tracing/trace.rmiss.glsl

APPS=\
	coherent_hashing \
	path_tracer \
	test_compute \
	test_mesh_shader \
	test_ray_tracing

SRC_DIR=shaders
BIN_DIR=spv/bin
LISTING_DIR=spv/disasm

DIRS=$(APPS:%=$(BIN_DIR)/%) $(APPS:%=$(LISTING_DIR)/%)
INCLUDES=$(INC:%=$(SRC_DIR)/%)
SHADERS=$(SRC:%.glsl=$(BIN_DIR)/%.spv)
LISTINGS=$(SRC:%.glsl=$(LISTING_DIR)/%.spv.txt)

$(info $(shell mkdir -p $(DIRS)))

all: shaders listings
.PHONY: all clean shaders listings images

clean:
	$(RM) $(SHADERS) $(LISTINGS)

shaders: $(SHADERS)

$(BIN_DIR)/%.rahit.spv: $(SRC_DIR)/%.rahit.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rcall.spv: $(SRC_DIR)/%.rcall.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rchit.spv: $(SRC_DIR)/%.rchit.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rgen.spv: $(SRC_DIR)/%.rgen.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rint.spv: $(SRC_DIR)/%.rint.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rmiss.spv: $(SRC_DIR)/%.rmiss.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.task.spv: $(SRC_DIR)/%.task.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_MESH) -o $@ $<

$(BIN_DIR)/%.spv: $(SRC_DIR)/%.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) -o $@ $<

listings: $(LISTINGS)

$(LISTING_DIR)/%.spv.txt: $(BIN_DIR)/%.spv Makefile
	$(DISASM) -o $@ $<

IMAGE_DIR=caldera/examples/path_tracer/images

IMAGES=\
	$(IMAGE_DIR)/bathroom2.jpg \
	$(IMAGE_DIR)/coffee.jpg \
	$(IMAGE_DIR)/glass-of-water.jpg \
	$(IMAGE_DIR)/staircase.jpg \
	$(IMAGE_DIR)/living-room-2.jpg \
	$(IMAGE_DIR)/staircase2.jpg \
	$(IMAGE_DIR)/spaceship.jpg \
	$(IMAGE_DIR)/cornell-box.jpg \
	$(IMAGE_DIR)/cornell-box_dome-light.jpg \
	$(IMAGE_DIR)/cornell-box_conductor.jpg \
	$(IMAGE_DIR)/cornell-box_conductor_surfaces-only.jpg \
	$(IMAGE_DIR)/cornell-box_conductor_lights-only.jpg \
	$(IMAGE_DIR)/material_conductors.jpg \
	$(IMAGE_DIR)/material_gold_f10_uniform.jpg \
	$(IMAGE_DIR)/material_gold_f10_hero.jpg \
	$(IMAGE_DIR)/material_gold_f10_continuous.jpg

clean-images:
	$(RM) $(IMAGES)

images: $(IMAGES)

MANY_SAMPLES=-s 10
PATH_TRACER=cargo run --release --example path_tracer --

$(IMAGE_DIR)/bathroom2.%: ../tungsten_scenes/bathroom2/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 560 -e -0.5 --fov 0.62 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/coffee.%: ../tungsten_scenes/coffee/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 492 -h 875 -e -0.5 -b 16 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/glass-of-water.%: ../tungsten_scenes/glass-of-water/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 560 -b 24 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/living-room-2.%: ../tungsten_scenes/living-room-2/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 560 -e -0.5 --fov 1.03 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/staircase.%: ../tungsten_scenes/staircase/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 492 -h 875 -e -0.5 -b 16 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/staircase2.%: ../tungsten_scenes/staircase2/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 1000 -e -0.5 -b 16 --planar-lights-are-two-sided enable $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/spaceship.%: ../tungsten_scenes/spaceship/scene.json shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 560 -b 16 $(MANY_SAMPLES) tungsten $<

$(IMAGE_DIR)/cornell-box.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 492 -h 492 -e 1.0 $(MANY_SAMPLES) cornell-box

$(IMAGE_DIR)/cornell-box_dome-light.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 492 -h 492 $(MANY_SAMPLES) cornell-box dome-light

$(IMAGE_DIR)/cornell-box_conductor.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 5 -f box cornell-box conductor

$(IMAGE_DIR)/cornell-box_conductor_surfaces-only.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 6 -f box --sampling-technique surfaces-only cornell-box conductor

$(IMAGE_DIR)/cornell-box_conductor_lights-only.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 6 -f box --sampling-technique lights-only cornell-box conductor

$(IMAGE_DIR)/material_conductors.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 1000 -h 560 $(MANY_SAMPLES) material-test ../ply/dragon_recon/dragon_vrip.ply conductors e

$(IMAGE_DIR)/material_gold_f10_uniform.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 3 --wavelength-sampling-method uniform material-test ../ply/dragon_recon/dragon_vrip.ply gold f10

$(IMAGE_DIR)/material_gold_f10_hero.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 3 --wavelength-sampling-method hero-mis material-test ../ply/dragon_recon/dragon_vrip.ply gold f10

$(IMAGE_DIR)/material_gold_f10_continuous.%: shaders Makefile
	$(PATH_TRACER) -o $@ -w 320 -h 320 -s 3 --wavelength-sampling-method continuous-mis material-test ../ply/dragon_recon/dragon_vrip.ply gold f10
