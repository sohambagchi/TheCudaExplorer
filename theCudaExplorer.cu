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
    printf("-m <string> : The type of memory to use (DRAM or UM)\n");
    printf("-c <string> : Memory Order (acq, rel, acq_rel, acq_acq)\n\n");

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
        printf("%s    (%ld ns)\t[%ld ns]\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? durations[j] / *count : (int64_t) (milliseconds[k] / *count), sequence[i].device == CE_CPU ? durations[j++] : (int64_t) (milliseconds[k]));
        printf("%s    (%u cycles)\t[%u ns]\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? 0 : (loopDurations[k] / *count), sequence[i].device == CE_CPU ? 0 : loopDurations[k++]);
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
    CEOrder gpuMemoryOrder;
    CEOrder cpuMemoryOrder;
    CECount outerLoopCount;
    int numObjects;    

    bool warmup = false;
    
    int opt;
    while ((opt = getopt(argc, argv, "n:o:m:c:g:l:wh")) != -1) {
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
            case 'g':
                if (strcmp(optarg, "acq") == 0) {
                    gpuMemoryOrder = CE_ACQ;
                } else if (strcmp(optarg, "rel") == 0) {
                    gpuMemoryOrder = CE_REL;
                } else if (strcmp(optarg, "acq-rel") == 0) {
                    gpuMemoryOrder = CE_ACQ_REL;
                } else if (strcmp(optarg, "acq-acq") == 0) {
                    gpuMemoryOrder = CE_ACQ_ACQ;
                } else if (strcmp(optarg, "none") == 0) {
                    gpuMemoryOrder = CE_NONE;
                } else {
                    printf("Invalid Memory Order: %s\n", optarg);
                    abort();
                }
                break;
            case 'c': 
                if (strcmp(optarg, "acq") == 0) {
                    cpuMemoryOrder = CE_ACQ;
                } else if (strcmp(optarg, "rel") == 0) {
                    cpuMemoryOrder = CE_REL;
                } else if (strcmp(optarg, "acq-rel") == 0) {
                    cpuMemoryOrder = CE_ACQ;
                } else if (strcmp(optarg, "acq-acq") == 0) {
                    cpuMemoryOrder = CE_ACQ;
                } else if (strcmp(optarg, "none") == 0) {
                    cpuMemoryOrder = CE_NONE;
                } else {
                    printf("Invalid Memory Order: %s\n", optarg);
                    abort();
                }
                break;
            case 'l':
                if (strcmp(optarg, "1K") == 0) {
                    outerLoopCount = CE_1K;
                } else if (strcmp(optarg, "10K") == 0) {
                    outerLoopCount = CE_10K;
                } else if (strcmp(optarg, "100K") == 0) {
                    outerLoopCount = CE_100K;
                } else if (strcmp(optarg, "1M") == 0) {
                    outerLoopCount = CE_1M;
                } else if (strcmp(optarg, "10M") == 0) {
                    outerLoopCount = CE_10M;
                } else if (strcmp(optarg, "100M") == 0) {
                    outerLoopCount = CE_100M;
                } else if (strcmp(optarg, "1B") == 0) {
                    outerLoopCount = CE_1B;
                } else if (strcmp(optarg, "1") == 0) {
                    outerLoopCount = CE_BASE;
                } else {
                    printf("Invalid Loop Count: %s\n", optarg);
                    abort();
                }
                break;
            case 'w':
                warmup = true;
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

    int GPUConsumeFirst = -1;

    // identify CPU and GPU events separately for timers
    for (int i = 0; i < operationSequence[0].total; i++) {
        // printf("%s %s\n", operationSequence[i].device == CE_CPU ? "CPU" : "\t\tGPU", operationSequence[i].action == CE_LOAD ? "ld" : "st");
        if (operationSequence[i].device == CE_CPU) {
            numCPUEvents++;
        } else {
            numGPUEvents++;
            if (GPUConsumeFirst == -1 && operationSequence[i].action == CE_LOAD) {
                GPUConsumeFirst = 1;
            } else if (GPUConsumeFirst == -1 && operationSequence[i].action == CE_STORE) {
                GPUConsumeFirst = 0;
            }
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

    printf("Size of Object: %.2f MB, %.2f KB\n", sizeof(struct LargeObject) / (1024.0 * 1024.0), sizeof(struct LargeObject) / 1024.0);
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

    printf("\nUsing %s for Objects\n", memoryType == CE_DRAM ? "cudaMallocHost" : memoryType == CE_UM ? "cudaMallocManaged" : "cudaMalloc");
    printf("Per-Iteration Loads: %s\n\n", 
    gpuMemoryOrder == CE_ACQ ? "acquire" : gpuMemoryOrder == CE_REL ? "release" : gpuMemoryOrder == CE_ACQ_ACQ ? "acq/acq" : gpuMemoryOrder == CE_ACQ_REL ? "acq/rel" : "non-atomic");


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

    if (warmup && GPUConsumeFirst == 1) {
        printf("Warming up GPU\n");
        switch (gpuMemoryOrder) {
            case CE_ACQ:
                GPUListConsumer_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_REL:
                GPUListConsumer_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_ACQ_ACQ:
                GPUListConsumer_acq_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_ACQ_REL:
                GPUListConsumer_acq_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_NONE:
                GPUListConsumer<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
        }
        cudaDeviceSynchronize();
    }

    // do all the operations in the order specified
    for (int i = 0; i < operationSequence[0].total; i++) {
        if (operationSequence[i].device == CE_CPU) {
            begin[CPUEventCount] = std::chrono::high_resolution_clock::now();
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    switch (cpuMemoryOrder) {
                        case CE_ACQ:
                            CPUListConsumer_acq(flag, largeObjectList, localConsumer, localOrder, count);
                            break;
                        case CE_REL:
                            CPUListConsumer_rel(flag, largeObjectList, localConsumer, localOrder, count);
                            break;
                        case CE_NONE:
                            CPUListConsumer(flag, largeObjectList, localConsumer, localOrder, count);
                            break;
                    }
                    break;
                case CE_STORE:
                    switch (cpuMemoryOrder) {
                        case CE_NONE:
                            CPUListProducer(flag, largeObjectList, localOrder, count);
                            break;
                        case CE_ACQ:
                        case CE_REL:
                            CPUListProducer_rel(flag, largeObjectList, localOrder, count);
                            break;
                    }
                    break;
            }
            end[CPUEventCount++] = std::chrono::high_resolution_clock::now();
        } else {
            cudaEventRecord(start[GPUEventCount]);
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    switch (gpuMemoryOrder) {
                        case CE_ACQ:
                            switch (outerLoopCount) {
                                case CE_1K:
                                    GPUListConsumer_acq_1K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10K:
                                    GPUListConsumer_acq_10K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100K:
                                    GPUListConsumer_acq_100K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;

                                case CE_1M:
                                    GPUListConsumer_acq_1M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10M:
                                    GPUListConsumer_acq_10M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100M:
                                    GPUListConsumer_acq_100M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;

                                case CE_1B:
                                    GPUListConsumer_acq_1B<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;

                                case CE_BASE:
                                    GPUListConsumer_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                            }
                            break;
                        case CE_REL:
                            switch (outerLoopCount) {
                                case CE_1K:
                                    GPUListConsumer_rel_1K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10K:
                                    GPUListConsumer_rel_10K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100K:
                                    GPUListConsumer_rel_100K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1M:
                                    GPUListConsumer_rel_1M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10M:
                                    GPUListConsumer_rel_10M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100M:
                                    GPUListConsumer_rel_100M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1B:
                                    GPUListConsumer_rel_1B<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_BASE:
                                    GPUListConsumer_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                            }
                            break;
                        case CE_ACQ_ACQ:
                            switch (outerLoopCount) {
                                case CE_1K:
                                    GPUListConsumer_acq_acq_1K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10K:
                                    GPUListConsumer_acq_acq_10K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100K:
                                    GPUListConsumer_acq_acq_100K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1M:
                                    GPUListConsumer_acq_acq_1M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10M:
                                    GPUListConsumer_acq_acq_10M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100M:
                                    GPUListConsumer_acq_acq_100M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1B:
                                    GPUListConsumer_acq_acq_1B<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_BASE:
                                    GPUListConsumer_acq_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                            }
                            break;
                        case CE_ACQ_REL:
                            switch (outerLoopCount) {
                                case CE_1K:
                                    GPUListConsumer_acq_rel_1K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10K:
                                    GPUListConsumer_acq_rel_10K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100K:
                                    GPUListConsumer_acq_rel_100K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1M:
                                    GPUListConsumer_acq_rel_1M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10M:
                                    GPUListConsumer_acq_rel_10M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100M:
                                    GPUListConsumer_acq_rel_100M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1B:
                                    GPUListConsumer_acq_rel_1B<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_BASE:
                                    GPUListConsumer_acq_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                            }
                            break;
                        case CE_NONE:
                            switch (outerLoopCount) {
                                case CE_1K:
                                    GPUListConsumer_1K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10K:
                                    GPUListConsumer_10K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100K:
                                    GPUListConsumer_100K<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1M:
                                    GPUListConsumer_1M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_10M:
                                    GPUListConsumer_10M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_100M:
                                    GPUListConsumer_100M<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_1B:
                                    GPUListConsumer_1B<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                                case CE_BASE:
                                    GPUListConsumer<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                                    break;
                            }
                            break;
                    }
                    SAFE(cudaMemcpy(&localBeforeLoop[GPUEventCount], beforeLoop[GPUEventCount], sizeof(unsigned int), cudaMemcpyDeviceToHost));
                    break;
                case CE_STORE:
                    switch (gpuMemoryOrder) {
                        case CE_NONE:
                            EmptyKernel<<<1,1>>>(count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                            // GPUListProducer<<<1,1>>>(flag, largeObjectList, largeObjectListOrder, count);
                            break;
                        case CE_ACQ:
                        case CE_REL:
                            GPUListProducer_rel<<<1,1>>>(flag, largeObjectList, largeObjectListOrder, count);
                            break;
                    }
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
}
