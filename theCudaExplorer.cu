#include <cstdio>
#include <chrono>

#include "theCudaExplorer.cuh"

char randString[PADDING_LENGTH];

// creates a random string for the object padding
void initRandString(int paddingSize) {
    for (int i = 0; i < paddingSize; i++) {
        randString[i] = rand() % 26 + 'a';
    }
}

// creates the shuffling order for the objects
void shuffleList(int * localOrder, int count) {
    for (int i = 0; i < count; i++) {
        int j = rand() % count;
        int tmp = localOrder[i];
        localOrder[i] = localOrder[j];
        localOrder[j] = tmp;
    }
}

#define SAFE(x) if (0 != x) { abort(); }

void printHelp() {
    printf("-n <int> : Number of Objects\n");
    printf("-o <string> : The order of operations\n");
    printf("-m <string> : The type of memory to use (DRAM or UM)\n\n");

    printf("Example: ./theCudaExplorer -n 1024 -o \"PcCgg\" -m DRAM\n");
    printf("This runs the following Litmus Test using cudaMallocHost\n\n");
    printf("CPU st\n\t\tGPU ld\n\t\tGPU ld\n\n");
}

CEOperation * parseOperations(const char* operations) { 

    int total_operations = 0;

    for (int i = 0; i < strlen(operations); i++) {
        if (operations[i] == 'c' || operations[i] == 'g') total_operations++;
    }

    CEOperation * sequence = (CEOperation *) malloc(sizeof(CEOperation) * total_operations);

    CEAction mode;
    for (int i = 0, j = 0; i < strlen(operations), j < total_operations; i++) {
        
        switch (operations[i]) {
            case 'P':
                mode = CE_STORE;
                break;
            case 'C':
                mode = CE_LOAD;
                break;
            case 'g':
                sequence[j].total = total_operations;
                sequence[j].device = CE_GPU;
                sequence[j++].action = mode;
                break;
            case 'c':
                sequence[j].total = total_operations;
                sequence[j].device = CE_CPU;
                sequence[j++].action = mode;
                break;
            default:
                printf("Invalid Operation: %c\n", operations[i]);
                abort();
        }
    }
    return sequence;
}

void printSequence(CEOperation * sequence) {

    printf("Sequence\n======================\n");

    for (int i = 0; i < sequence[0].total; i++) {
        printf("%s %s\n", sequence[i].device == CE_CPU ? "CPU" : "\t\tGPU", sequence[i].action == CE_LOAD ? "ld" : "st");
    }

    printf("\n");

}

void printResults(CEOperation * sequence, int numCPUEvents, int numGPUEvents, int64_t * durations, unsigned int * loopDurations, float * milliseconds, int * count) {
    
    printf("Results\n======================\n");

    for (int i = 0, j = 0, k = 0; i < sequence[0].total; i++) {
        printf("%s %s\n", sequence[i].device == CE_CPU ? "CPU" : "\t\tGPU", sequence[i].action == CE_LOAD ? "ld" : "st");
        printf("%s    (%ld ns)\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? durations[j++] / *count : (int64_t) (milliseconds[k] / *count));
        printf("%s    (%u cycles)\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? 0 : (loopDurations[k++] / *count));
    }

}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        printHelp();
        return 0;
    }

    CEMemory memoryType;
    CEOperation * operationSequence;
    int numObjects;
    
        int opt;
    while ((opt = getopt(argc, argv, "n:o:m:h")) != -1) {
        switch (opt) {
            case 'n':
                numObjects = atoi(optarg);
                break;
            case 'o':
                operationSequence = parseOperations(optarg);
                break;
            case 'm':
                if (strcmp(optarg, "DRAM") == 0) {
                    memoryType = CE_DRAM;
                } else if (strcmp(optarg, "UM") == 0) {
                    memoryType = CE_UM;
                } else if (strcmp(optarg, "GDDR") == 0) {
                    memoryType = CE_GDDR;
                } else {
                    printf("Invalid Memory Type: %s\n", optarg);
                    abort();
                }
                break;
            case 'h':
                printHelp();
                return 0;
            default:
                printHelp();
                return 0;
        }
    }

    int numCPUEvents = 0, numGPUEvents = 0;

    // identify CPU and GPU events separately for timers
    for (int i = 0; i < operationSequence[0].total; i++) {
        // printf("%s %s\n", operationSequence[i].device == CE_CPU ? "CPU" : "\t\tGPU", operationSequence[i].action == CE_LOAD ? "ld" : "st");
        if (operationSequence[i].device == CE_CPU) {
            numCPUEvents++;
        } else {
            numGPUEvents++;
        }
    }

    // CPU Timers
    std::chrono::high_resolution_clock::time_point begin[numCPUEvents], end[numCPUEvents];
    int64_t durations[numCPUEvents];

    // CUDA Timers
    cudaEvent_t start[numGPUEvents], stop[numGPUEvents];
    float milliseconds[numGPUEvents];

    // Internal clock64() timers
    unsigned int * beforeLoop[numGPUEvents];
    unsigned int * afterLoop[numGPUEvents];
    unsigned int localBeforeLoop[numGPUEvents];
    unsigned int localAfterLoop[numGPUEvents];
    unsigned int loopDuration[numGPUEvents];

    for (int i = 0; i < numGPUEvents; i++) {
        SAFE(cudaEventCreate(&start[i]));
        SAFE(cudaEventCreate(&stop[i]));
        SAFE(cudaMalloc(&beforeLoop[i], sizeof(unsigned int)));
        SAFE(cudaMalloc(&afterLoop[i], sizeof(unsigned int)));
    }

    cuda::atomic<int>* flag;
    struct LargeObject * largeObjectList;
    int * largeObjectListConsumer;
    int * localConsumer;
    int * largeObjectListOrder;
    int * localOrder;
    int * count;

    SAFE(cudaMallocHost(&count, sizeof(int)));

    *count = numObjects;

    printf("Size of Object: %.2f MB\n", sizeof(struct LargeObject) / (1024.0 * 1024.0));
    printf("Number of Objects: %d\n", *count);
    printf("CPU Events Timed: %d\t GPU Events Timed: %d\n", numCPUEvents, numGPUEvents);

    SAFE(cudaMallocHost(&flag, sizeof(cuda::atomic<int>)));
    SAFE(cudaMallocHost(&largeObjectListConsumer, sizeof(int) * *count));
    SAFE(cudaMallocHost(&localConsumer, sizeof(int) * *count));
    SAFE(cudaMallocHost(&largeObjectListOrder, sizeof(int) * *count));
    SAFE(cudaMallocHost(&localOrder, sizeof(int) * *count));

    if (memoryType == CE_DRAM) {
        SAFE(cudaMallocHost(&largeObjectList, sizeof(struct LargeObject) * *count));
    } else if (memoryType == CE_UM) {
        SAFE(cudaMallocManaged(&largeObjectList, sizeof(struct LargeObject) * *count));
    } else {
        SAFE(cudaMalloc(&largeObjectList, sizeof(struct LargeObject) * *count));
    }
        
    // allocate the ordering array in both DRAM and GDDR
    for (int i = 0; i < (*count); i++) {
        localOrder[i] = i;
    }

    shuffleList(localOrder, *count);
    SAFE(cudaMemcpy(largeObjectListOrder, localOrder, sizeof(int) * *count, cudaMemcpyHostToDevice));

    printf("\nUsing %s for Objects\n\n", memoryType == CE_DRAM ? "cudaMallocHost" : memoryType == CE_UM ? "cudaMallocManaged" : "cudaMalloc");

    //randomly pad all the objects
    
    if (memoryType != CE_GDDR) {
        for (int i = 0; i < (*count); i++) {
            initRandString((sizeof(LargeObject) - sizeof(int)) / (2 * sizeof(char)));
            strcpy(largeObjectList[localOrder[i]].padding1, randString);
            // initRandString((sizeof(LargeObject) - sizeof(int)) / (2 * sizeof(char)));
            strcpy(largeObjectList[localOrder[i]].padding2, randString);
            // (largeObjectList[localOrder[i]]).data = i;
            (largeObjectList[localOrder[i]]).data.store(i);
        }
    }

    printSequence(operationSequence);

    int CPUEventCount = 0;
    int GPUEventCount = 0;

    // do all the operations in the order specified
    for (int i = 0; i < operationSequence[0].total; i++) {
        if (operationSequence[i].device == CE_CPU) {
            begin[CPUEventCount] = std::chrono::high_resolution_clock::now();
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    CPUListConsumer(flag, largeObjectList, localConsumer, localOrder, count);
                    break;
                case CE_STORE:
                    CPUListProducer(flag, largeObjectList, localOrder, count);
                    break;
            }
            end[CPUEventCount++] = std::chrono::high_resolution_clock::now();
        } else {
            cudaEventRecord(start[GPUEventCount]);
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    GPUListConsumer<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                    SAFE(cudaMemcpy(&localBeforeLoop[GPUEventCount], beforeLoop[GPUEventCount], sizeof(unsigned int), cudaMemcpyDeviceToHost));
                    break;
                case CE_STORE:
                    GPUListProducer<<<1,1>>>(flag, largeObjectList, largeObjectListOrder, count);
                    break;
            }
            cudaEventRecord(stop[GPUEventCount]);

            // synchronize GPU executation after every operation
            cudaEventSynchronize(stop[GPUEventCount]);
            GPUEventCount++;
        }
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < numGPUEvents; i++) {
        milliseconds[i] = 0;
        cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
        milliseconds[i] *= 1e6;
        loopDuration[i] = localAfterLoop[i] - localBeforeLoop[i];
    }

    for (int i = 0; i < numCPUEvents; i++) {
        durations[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end[i] - begin[i]).count();
    }

    printResults(operationSequence, numCPUEvents, numGPUEvents, durations, loopDuration, milliseconds, count);

    printf("\n----------------------\n\n\n");

    return 0;


    // LABEL:MH40
    // struct LargeObject ** remoteObjectList;
    
    // SAFE(cudaMallocHost(&remoteObjectList, sizeof(struct LargeObject*) * *count));
      
    // for (int i = 0; i < (*count); i++) {
    //     SAFE(cudaMalloc(&remoteObjectList[*localOrder[i]], sizeof(struct LargeObject)));
    // }

    // SAFE(cudaMemcpy(remoteObjectList[*localOrder[i]], largeObjectList[*localOrder[i]], sizeof(struct LargeObject), cudaMemcpyHostToDevice));

}
