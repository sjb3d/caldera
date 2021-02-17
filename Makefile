GLSLC=glslangValidator
GLSLCFLAGS_COMMON=-V -Ishaders/common
GLSLCFLAGS_RAYS=--target-env spirv1.4

DISASM=spirv-dis

INC=\
	common/color_space.glsl \
	common/fresnel.glsl \
	common/maths.glsl \
	common/normal_pack.glsl \
	common/sampler.glsl \
	common/ggx.glsl \
	trace/bsdf_common.glsl \
	trace/diffuse_bsdf.glsl \
	trace/dome_light.glsl \
	trace/extend_common.glsl \
	trace/light_common.glsl \
	trace/mirror_bsdf.glsl \
	trace/occlusion_common.glsl \
	trace/quad_light.glsl \
	trace/rand_common.glsl \
	trace/rough_conductor_bsdf.glsl \
	trace/rough_plastic_bsdf.glsl \
	trace/solid_angle_light.glsl \
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
	trace/copy.frag.glsl \
	trace/copy.vert.glsl \
	trace/extend.rmiss.glsl \
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
.PHONY: all clean shaders listings

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
