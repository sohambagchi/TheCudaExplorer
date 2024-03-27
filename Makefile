NVCC = nvcc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O0 -arch=sm_80
SOURCES = theCudaExplorer.cu
HEADERS = theCudaExplorer.cuh
PTX = theCudaExplorer.ptx
OUTPUT = theCudaExplorer


all: $(OUTPUT) ptx

$(OUTPUT): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $(OUTPUT) $(SOURCES)

ptx: $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -ptx -src-in-ptx $(SOURCES)

clean:
	rm -f $(OUTPUT) $(PTX)