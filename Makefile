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

SIZES =      12  28  60   92   124  188  252 508 1020 2044 4092 8188 16380 32764 65532 131068 262140 524284 1048572 2097148 4194300 8388604 16777212 33554428 67108860 134217724
# PAYLOAD = 32B 64B 128B 192B 256B 384B 512B 1KB  2KB  4KB 8KB  16KB  32KB  64KB 128KB  256KB 512KB   1MB     2MB     4MB     8MB     16MB    32MB     64MB    128MB     256MB

# SIZES =  260   1284  2564 4092  8188 16380 32764 65532 131066 241664 479228 1048570 2097146 4194298 8388604 16777216
# SIZES =  16777216
# SIZES =64KB 320KB 640KB  1MB  2MB  4MB   8MB   16MB  32MB   60MB   117MB  256MB   512MB    1GB      2GB     4GB 
# SIZES = 479228

all: $(OUTPUT) 

$(OUTPUT): $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for size in $(SIZES); do \
			$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DPADDING_LENGTH=$$size -o $(OUTPUT)_$$scope-$$size.out $(SOURCES); \
		done; \
	done
	python3 cleanup_executables.py clean

ptx: $(SOURCES) $(HEADERS)
	for scope in $(SCOPES); do \
		for size in $(SIZES); do \
			$(NVCC) $(NVCCFLAGS) -DSCOPE=$$scope -DPADDING_LENGTH=$$size -ptx -src-in-ptx $(SOURCES); \
			cp $(PTX) $(PTX)_$$scope-$$size; \
			$(CUOBJDUMP) $(CUOBJDUMPFLAGS) $(OUTPUT)_$$scope-$$size.out > $(SASS)_$$scope-$$size; \
		done; \
	done
	python3 cleanup_executables.py clean

clean:
	rm -f $(OUTPUT) $(PTX) $(SASS)
	rm -f *.out *.ptx *.sass
