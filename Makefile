GLSLC=glslangValidator
GLSLCFLAGS_COMMON=-V -Ishaders/common
GLSLCFLAGS_RAYS=--target-env spirv1.4

DISASM=spirv-dis

INC=\
	common/color_space.glsl \
	common/normal_pack.glsl \
	common/sampler.glsl

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
	mesh/trace.rmiss.glsl

APPS=compute mesh
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

$(BIN_DIR)/%.rchit.spv: $(SRC_DIR)/%.rchit.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rgen.spv: $(SRC_DIR)/%.rgen.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.rmiss.spv: $(SRC_DIR)/%.rmiss.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) $(GLSLCFLAGS_RAYS) -o $@ $<

$(BIN_DIR)/%.spv: $(SRC_DIR)/%.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS_COMMON) -o $@ $<

listings: $(LISTINGS)

$(LISTING_DIR)/%.spv.txt: $(BIN_DIR)/%.spv Makefile
	$(DISASM) -o $@ $<
