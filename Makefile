NVCC = nvc++
NVCCFLAGS = -g -gpu=mem:unified:nomanagedalloc
# NVCCFLAGS = -arch=sm_80
SOURCES = theCudaExplorer.cu
HEADERS = theCudaExplorer.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer
CUOBJDUMP = cuobjdump
CUOBJDUMPFLAGS = -sass
SASS = theCudaExplorer.sass

SCOPES = cuda::thread_scope_thread cuda::thread_scope_block cuda::thread_scope_device cuda::thread_scope_system
# LOAD_ORDER = cuda::memory_order_acquire cuda::memory_order_relaxed

all: $(OUTPUT) #ptx

$(OUTPUT): $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -o $(OUTPUT)_$$scope.out $(SOURCES); \
	done
# $(NVCC) $(NVCCFLAGS)  -o $(OUTPUT) $(SOURCES)

ptx: $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -ptx -src-in-ptx $(SOURCES); \
		cp $(PTX) $(PTX)_$$scope-87; \
		$(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT)_$$scope.out > $(SASS)_$$scope-87; \
	done
# $(NVCC) $(NVCCFLAGS) -ptx -src-in-ptx $(SOURCES)
# $(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT) > $(SASS)


clean:
	rm -f $(OUTPUT) $(PTX) $(SASS)
	rm -f *.out *.ptx *.sass