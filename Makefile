GLSLC=glslangValidator
GLSLCFLAGS_COMMON=-V -Ishaders/common
GLSLCFLAGS_RAYS=--target-env spirv1.4

DISASM=spirv-dis

INC=\
	common/color_space.glsl \
	common/fresnel.glsl \
	common/ggx.glsl \
	common/maths.glsl \
	common/normal_pack.glsl \
	common/sampler.glsl \
	common/tone_map.glsl \
	trace/bsdf_common.glsl \
	trace/diffuse_bsdf.glsl \
	trace/disc_common.glsl \
	trace/disc_light.glsl \
	trace/dome_light.glsl \
	trace/extend_common.glsl \
	trace/light_common.glsl \
	trace/mirror_bsdf.glsl \
	trace/occlusion_common.glsl \
	trace/quad_light.glsl \
	trace/rough_conductor_bsdf.glsl \
	trace/rough_dielectric_bsdf.glsl \
	trace/rough_plastic_bsdf.glsl \
	trace/sequence.glsl \
	trace/solid_angle_light.glsl \
	trace/spectrum.glsl \
	trace/sphere_common.glsl \
	trace/sphere_light.glsl \
	trace/smooth_dielectric_bsdf.glsl \
	trace/smooth_plastic_bsdf.glsl

SRC=\
	compute/trace.comp.glsl \
	compute/copy.frag.glsl \
	compute/copy.vert.glsl \
	mesh/copy.frag.glsl \
	mesh/copy.vert.glsl \
	mesh/raster.vert.glsl \
	mesh/raster.frag.glsl \
	mesh/trace.rchit.glsl \
	mesh/trace.rgen.glsl \
	mesh/trace.rmiss.glsl \
	trace/capture.comp.glsl \
	trace/copy.frag.glsl \
	trace/copy.vert.glsl \
	trace/disc.rint.glsl \
	trace/extend.rmiss.glsl \
	trace/extend_disc.rchit.glsl \
	trace/extend_sphere.rchit.glsl \
	trace/extend_triangle.rchit.glsl \
	trace/filter.comp.glsl \
	trace/occlusion.rchit.glsl \
	trace/occlusion.rmiss.glsl \
	trace/path_trace.rgen.glsl \
	trace/sphere.rint.glsl \

APPS=compute mesh trace

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

$(BIN_DIR)/%.spv: $(SRC_DIR)/%.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) -o $@ $<

listings: $(LISTINGS)

$(LISTING_DIR)/%.spv.txt: $(BIN_DIR)/%.spv Makefile
	$(DISASM) -o $@ $<

IMAGES=\
	docs/trace_bathroom2.jpg \
	docs/trace_coffee.jpg \
	docs/trace_glass-of-water.jpg \
	docs/trace_staircase.jpg \
	docs/trace_living-room-2.jpg \
	docs/trace_staircase2.jpg \
	docs/trace_cornell-box.jpg \
	docs/trace_cornell-box_dome-light.jpg \
	docs/trace_cornell-box_conductor.jpg \
	docs/trace_cornell-box_conductor_surfaces-only.jpg \
	docs/trace_cornell-box_conductor_lights-only.jpg \
	docs/trace_material_conductors.jpg

clean-images:
	$(RM) $(IMAGES)

images: $(IMAGES)

MANY_SAMPLES=-s 10
TRACE=cargo run --release --bin trace --

docs/trace_bathroom2.%: ../tungsten_scenes/bathroom2/scene.json shaders Makefile
	$(TRACE) -o $@ -w 1000 -h 560 -e -0.5 --fov 0.62 $(MANY_SAMPLES) tungsten $<

docs/trace_coffee.%: ../tungsten_scenes/coffee/scene.json shaders Makefile
	$(TRACE) -o $@ -w 492 -h 875 -e -0.5 -b 16 $(MANY_SAMPLES) tungsten $<

docs/trace_glass-of-water.%: ../tungsten_scenes/glass-of-water/scene.json shaders Makefile
	$(TRACE) -o $@ -w 1000 -h 560 -b 24 $(MANY_SAMPLES) tungsten $<

docs/trace_living-room-2.%: ../tungsten_scenes/living-room-2/scene.json shaders Makefile
	$(TRACE) -o $@ -w 1000 -h 560 -e -0.5 --fov 1.03 $(MANY_SAMPLES) tungsten $<

docs/trace_staircase.%: ../tungsten_scenes/staircase/scene.json shaders Makefile
	$(TRACE) -o $@ -w 492 -h 875 -e -0.5 -b 16 $(MANY_SAMPLES) tungsten $<

docs/trace_staircase2.%: ../tungsten_scenes/staircase2/scene.json shaders Makefile
	$(TRACE) -o $@ -w 1000 -h 1000 -e -0.5 -b 16 --planar-lights-are-two-sided enable $(MANY_SAMPLES) tungsten $<

docs/trace_cornell-box.%: shaders Makefile
	$(TRACE) -o $@ -w 492 -h 492 -e 1.0 $(MANY_SAMPLES) cornell-box

docs/trace_cornell-box_dome-light.%: shaders Makefile
	$(TRACE) -o $@ -w 492 -h 492 $(MANY_SAMPLES) cornell-box dome-light

docs/trace_cornell-box_conductor.%: shaders Makefile
	$(TRACE) -o $@ -w 320 -h 320 -s 5 -f box cornell-box conductor

docs/trace_cornell-box_conductor_surfaces-only.%: shaders Makefile
	$(TRACE) -o $@ -w 320 -h 320 -s 6 -f box --sampling-technique surfaces-only cornell-box conductor

docs/trace_cornell-box_conductor_lights-only.%: shaders Makefile
	$(TRACE) -o $@ -w 320 -h 320 -s 6 -f box --sampling-technique lights-only cornell-box conductor

docs/trace_material_conductors.%: shaders Makefile
	$(TRACE) -o $@ -w 1000 -h 560 $(MANY_SAMPLES) material-test ../ply/dragon_recon/dragon_vrip.ply
