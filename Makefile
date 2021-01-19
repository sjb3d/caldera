GLSLC=glslangValidator
GLSLCFLAGS=-V -Ishaders/common

DISASM=spirv-dis

INC=\
	common/color_space.glsl \
	common/sampler.glsl

SRC=\
	compute/trace.comp.glsl \
	compute/copy.frag.glsl \
	compute/copy.vert.glsl \
	mesh/test.vert.glsl \
	mesh/test.frag.glsl

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

$(BIN_DIR)/%.spv: $(SRC_DIR)/%.glsl $(INCLUDES) Makefile
	$(GLSLC) $(GLSLCFLAGS) -o $@ $<

listings: $(LISTINGS)

$(LISTING_DIR)/%.spv.txt: $(BIN_DIR)/%.spv Makefile
	$(DISASM) -o $@ $<
