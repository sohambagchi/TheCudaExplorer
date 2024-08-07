NVCC = nvcc
# NVCCFLAGS = -g -gpu=mem:unified:nomanagedalloc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O3 -arch=sm_87 -I./include
SOURCES = theCudaExplorer.cu
HEADERS = ./include/theCudaExplorer_top.cuh ./include/theCudaExplorer_array.cuh ./include/theCudaExplorer_list.cuh ./include/theCudaExplorer_loaded.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer
CUOBJDUMP = cuobjdump
CUOBJDUMPFLAGS = -sass
SASS = theCudaExplorer.sass

SCOPES = cuda::thread_scope_system # cuda::thread_scope_thread cuda::thread_scope_block cuda::thread_scope_device

SIZES =      12  28  60   92   124  188  252 508 1020 2044 4092 8188 16380 32764 65532 131068 262140 524284 1048572 2097148 4194300 8388604 16777212 33554428 67108860 134217724
# PAYLOAD = 32B 64B 128B 192B 256B 384B 512B 1KB  2KB  4KB 8KB  16KB  32KB  64KB 128KB  256KB 512KB   1MB     2MB     4MB     8MB     16MB    32MB     64MB    128MB     256MB

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
			cp $(PTX) $(PTX)_$$scope-$$size.ptx; \
			$(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT)_$$scope-$$size.out > $(SASS)_$$scope-$$size.sass; \
		done; \
	done
	python3 cleanup_executables.py

clean:
	rm -f $(OUTPUT) $(PTX) $(SASS)
	rm -f *.out *.ptx *.sass
