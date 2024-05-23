NVCC = nvcc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O3 -arch=sm_80
# NVCCFLAGS = -arch=sm_80
SOURCES = theCudaExplorer.cu
HEADERS = theCudaExplorer.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer
CUOBJDUMP = cuobjdump
CUOBJDUMPFLAGS = -sass
SASS = theCudaExplorer.sass

SCOPES = cuda::thread_scope_thread cuda::thread_scope_block cuda::thread_scope_device cuda::thread_scope_system
LOAD_ORDER = cuda::memory_order_acquire cuda::memory_order_relaxed

all: $(OUTPUT) ptx

$(OUTPUT): $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for load_order1 in $(LOAD_ORDER); do \
			for load_order2 in $(LOAD_ORDER); do \
				$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DLOAD1=$$load_order1 -DLOAD2=$$load_order2 -o $(OUTPUT)_$$scope$$load_order1$$load_order2.out $(SOURCES); \
			done; \
		done; \
	done
# $(NVCC) $(NVCCFLAGS)  -o $(OUTPUT) $(SOURCES)

ptx: $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for load_order1 in $(LOAD_ORDER); do \
			for load_order2 in $(LOAD_ORDER); do \
				$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DLOAD1=$$load_order1 -DLOAD2=$$load_order2 -ptx -src-in-ptx $(SOURCES); \
				cp $(PTX) $(PTX)_$$scope$$load_order1$$load_order2; \
			done; \
		done; \
	done
# $(NVCC) $(NVCCFLAGS) -ptx -src-in-ptx $(SOURCES)
# $(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT) > $(SASS)


clean:
	rm -f $(OUTPUT) $(PTX) $(SASS)