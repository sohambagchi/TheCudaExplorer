NVCC = nvcc
NVCCFLAGS = -g -Xcompiler -O0 -Xcicc -O0 -arch=sm_80
SOURCES = theCudaExplorer.cu
HEADERS = theCudaExplorer.cuh
OUTPUT = theCudaExplorer

all: $(OUTPUT)

$(OUTPUT): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $(OUTPUT) $(SOURCES)

clean:
	rm -f $(OUTPUT)