NVCC = nvcc
# NVCCFLAGS = -g -gpu=mem:unified:nomanagedalloc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O3 -arch=sm_87
SOURCES = theCudaExplorer.cu
HEADERS = theCudaExplorer.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer
CUOBJDUMP = cuobjdump
CUOBJDUMPFLAGS = -sass
SASS = theCudaExplorer.sass

SCOPES = cuda::thread_scope_thread  cuda::thread_scope_system # cuda::thread_scope_block cuda::thread_scope_device
SIZES = 241664 479228 16777216

all: $(OUTPUT) ptx

$(OUTPUT): $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for size in $(SIZES); do \
			$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DPADDING_LENGTH=$$size -o $(OUTPUT)_$$scope-$$size.out $(SOURCES); \
		done; \
	done

ptx: $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for size in $(SIZES); do \
			$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DPADDING_LENGTH=$$size -ptx -src-in-ptx $(SOURCES); \
			cp $(PTX) $(PTX)_$$scope-$$size; \
			$(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT)_$$scope-$$size.out > $(SASS)_$$scope-$$size; \
		done; \
	done
	python3 cleanup_executables_1.py clean

clean:
	rm -f $(OUTPUT) $(PTX) $(SASS)
	rm -f *.out *.ptx *.sass
