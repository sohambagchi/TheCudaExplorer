NVCC = nvcc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O3 -arch=sm_87 -I./include
SOURCES = theCudaExplorer.cu
HEADERS = ./include/theCudaExplorer_top.cuh ./include/theCudaExplorer_array.cuh ./include/theCudaExplorer_list.cuh ./include/theCudaExplorer_loaded.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer
CUOBJDUMP = cuobjdump
CUOBJDUMPFLAGS = -sass
SASS = theCudaExplorer.sass

SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_thread #cuda\:\:thread_scope_block cuda\:\:thread_scope_device
SIZES = 12 28 60 92 124 188 252 508 1020 2044 4092 8188 16380 32764 65532 131068 262140 524284 1048572 2097148 4194300 8388604 16777212 33554428 67108860 134217724

# Generate all targets
all: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_$(scope)_$(size).out))

# Generate PTX targets
ptx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(PTX)_$(scope)_$(size).ptx))
        python3 cleanup_executables.py

# Use eval and value to dynamically create rules for each combination of SCOPE and SIZE
define make_target
$(OUTPUT)_$(1)_$(2).out: $(SOURCES) $(HEADERS)
        $$(NVCC) $$(NVCCFLAGS) -DSCOPE=$(1) -DPADDING_LENGTH=$(2) -o $$@ $$(SOURCES)

$(PTX)_$(1)_$(2).ptx: $(SOURCES) $(HEADERS)
        $$(NVCC) $$(NVCCFLAGS) -DSCOPE=$(1) -DPADDING_LENGTH=$(2) -ptx -src-in-ptx $$(SOURCES) -o $$@
        $$(CUOBJDUMP) $$(CUOBJDUMPFLAGS) $$(OUTPUT)_$(1)_$(2).out > $$(SASS)_$(1)_$(2).sass
endef

# Iterate over SCOPES and SIZES to generate rules
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target,$(scope),$(size)))))

clean:
        rm -f $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_$(scope)_$(size).out $(PTX)_$(scope)_$(size).ptx $(SASS)_$(scope)_$(size).sass))
        rm -f *.out *.ptx *.sass
