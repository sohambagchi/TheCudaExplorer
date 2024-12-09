#include <cstdio>
#include <chrono>

#include "include/theCudaExplorer_top.cuh"
#include "include/theCudaExplorer_array.cuh"
#include "include/theCudaExplorer_list.cuh"

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

#define SAFE(x) if (0 != x) { printf("Error: %d @ Line %d\n", x, __LINE__); abort(); }

void printHelp() {
    printf("-n <int> : Number of Objects\n");
    printf("-o <string> : The order of operations\n");
    printf("-m <string> : The type of memory to use (GDDR, UM, DRAM or SYS)\n");
    printf("-c <string> : Memory Order (acq, rel, acq_rel, acq_acq, none) for CPU Operations\n");
    printf("-g <string> : Memory Order (acq, rel, acq_rel, acq_acq, none) for GPU Operations\n");
    printf("-l <1, 1K, 10K> : The number of outer-loop iterations \n");
    printf("-t <string> : The type of Array being used (array, linkedlist, loaded)\n");
    printf("-w : Include if warmup required\n");

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
        printf("%s    (%ld ns)\t[%ld ns]\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? (durations[j] / *count) : (int64_t) (milliseconds[k] / *count), sequence[i].device == CE_CPU ? durations[j] : (int64_t) (milliseconds[k]));
        printf("%s    (%u cycles)\t[%u ns]\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? 0 : (loopDurations[k] / *count), sequence[i].device == CE_CPU ? 0 : loopDurations[k]);
        if (sequence[i].device == CE_CPU) j++;
        else k++;
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
    CEObjectType objectType;
    int numObjects;

    bool warmup = false;
    
    int opt;
    while ((opt = getopt(argc, argv, "n:o:m:c:g:l:t:wh")) != -1) {
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
                } else if (strcmp(optarg, "SYS") == 0) {
                    memoryType = CE_SYS;
                } else if (strcmp(optarg, "NUMA_HOST") == 0) {
                    memoryType = CE_NUMA_HOST;
                } else if (strcmp(optarg, "NUMA_DEV") == 0) {
                    memoryType = CE_NUMA_DEVICE;
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
            case 't':
                if (strcmp(optarg, "array") == 0) {
                    objectType = CE_ARRAY;
                } else if (strcmp(optarg, "linkedlist") == 0) {
                    objectType = CE_LINKEDLIST;
                } else if (strcmp(optarg, "loaded") == 0) {
                    objectType = CE_LOADED;
                } else {
                    printf("Invalid Array Type: %s\n", optarg);
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
    unsigned int * localBeforeLoop[numGPUEvents];
    unsigned int * localAfterLoop[numGPUEvents];
    unsigned int loopDuration[numGPUEvents];

    for (int i = 0; i < numGPUEvents; i++) {
        SAFE(cudaEventCreate(&start[i]));
        SAFE(cudaEventCreate(&stop[i]));
        SAFE(cudaMalloc(&beforeLoop[i], sizeof(unsigned int)));
        SAFE(cudaMalloc(&afterLoop[i], sizeof(unsigned int)));

        localBeforeLoop[i] = (unsigned int *) calloc(1, sizeof(unsigned int));

        if (localBeforeLoop[i] == NULL) {
            printf("Failed to allocate memory for localBeforeLoop\n");
            abort();
        }

        localAfterLoop[i] = (unsigned int *) calloc(1, sizeof(unsigned int));

        if (localAfterLoop[i] == NULL) {
            printf("Failed to allocate memory for localAfterLoop\n");
            abort();
        }
    }

    cuda::atomic<int>* flag;
    struct LargeLinkedObject * largeObjectList;
    struct LargeLinkedObject * localCopy;
    int * largeObjectListConsumer;
    int * loadedListConsumer;
    int * localConsumer;
    int * localLoadedConsumer;
    int * largeObjectListOrder;
    int * localOrder;
    int * count;

    SAFE(cudaMallocHost(&count, sizeof(int)));

    *count = numObjects;

    printf("Size of Object: %.2f MB, %.2f KB\n", sizeof(struct LargeLinkedObject) / (1024.0 * 1024.0), sizeof(struct LargeLinkedObject) / 1024.0);
    printf("Number of Objects: %d\n", *count);
    printf("CPU Events Timed: %d\t GPU Events Timed: %d\n", numCPUEvents, numGPUEvents);

    if (memoryType == CE_SYS) {
        flag = (cuda::atomic<int> *) malloc(sizeof(cuda::atomic<int>));
        localConsumer = (int *) malloc(sizeof(int) * *count);
        localLoadedConsumer = (int *) malloc(sizeof(int) * *count);
        largeObjectListOrder = (int *) malloc(sizeof(int) * *count);
        localOrder = (int *) malloc(sizeof(int) * *count);
    } else if (memoryType == CE_NUMA_DEVICE || memoryType == CE_NUMA_HOST) {
        int numa_node = memoryType == CE_NUMA_HOST ? 0 : 1;

        flag = (cuda::atomic<int> *) numa_alloc_onnode(sizeof(cuda::atomic<int>), numa_node);
        localConsumer = (int *) numa_alloc_onnode(sizeof(int) * *count, numa_node);
        localLoadedConsumer = (int *) numa_alloc_onnode(sizeof(int) * *count, numa_node);
        largeObjectListOrder = (int *) numa_alloc_onnode(sizeof(int) * *count, numa_node);
        localOrder = (int *) numa_alloc_onnode(sizeof(int) * *count, numa_node);
    } else {
        SAFE(cudaMallocHost(&flag, sizeof(cuda::atomic<int>)));
        SAFE(cudaMallocHost(&localConsumer, sizeof(int) * *count));
        SAFE(cudaMallocHost(&localLoadedConsumer, sizeof(int) * *count));
        SAFE(cudaMallocHost(&largeObjectListOrder, sizeof(int) * *count));
        SAFE(cudaMallocHost(&localOrder, sizeof(int) * *count));
    }

    if (memoryType == CE_GDDR) {
        SAFE(cudaMalloc(&largeObjectListConsumer, sizeof(int) * *count));
        SAFE(cudaMalloc(&loadedListConsumer, sizeof(int) * *count));
    } else {
        SAFE(cudaMallocHost(&largeObjectListConsumer, sizeof(int) * *count));
        SAFE(cudaMallocHost(&loadedListConsumer, sizeof(int) * *count));
    }

    if (memoryType == CE_DRAM) {
        SAFE(cudaMallocHost(&largeObjectList, sizeof(struct LargeLinkedObject) * *count));
    } else if (memoryType == CE_UM) {
        SAFE(cudaMallocManaged(&largeObjectList, sizeof(struct LargeLinkedObject) * *count));
    } else if (memoryType == CE_GDDR) {
        SAFE(cudaMalloc(&largeObjectList, sizeof(struct LargeLinkedObject) * *count));
    } else if (memoryType == CE_SYS) {
        largeObjectList = (struct LargeLinkedObject*) malloc(sizeof(struct LargeLinkedObject) * *count);
    } else if (memoryType == CE_NUMA_DEVICE || memoryType == CE_NUMA_HOST) {
        int numa_node = memoryType == CE_NUMA_HOST ? 0 : 1;
        largeObjectList = (struct LargeLinkedObject*) numa_alloc_onnode(sizeof(struct LargeLinkedObject) * *count, numa_node);
    }

    localCopy = (struct LargeLinkedObject *) malloc(sizeof(struct LargeLinkedObject) * *count);

    // allocate the ordering array in both DRAM and GDDR
    for (int i = 0; i < (*count); i++) {
        localOrder[i] = i;
    }

    shuffleList(localOrder, *count);
    SAFE(cudaMemcpy(largeObjectListOrder, localOrder, sizeof(int) * *count, cudaMemcpyHostToDevice));

    printf("\nUsing %s for %s Objects\n", memoryType == CE_DRAM ? "cudaMallocHost" : memoryType == CE_UM ? "cudaMallocManaged" : memoryType == CE_GDDR ? "cudaMalloc" : memoryType == CE_SYS ? "malloc" : memoryType == CE_NUMA_HOST ? "numa_alloc_onnode(host)" : "numa_alloc_onnode(device)", objectType == CE_ARRAY ? "Array" : objectType == CE_LINKEDLIST ? "LinkedList" : "Loaded List");
    printf("Per-Iteration Loads: %s (%s iter)\n\n", 
    gpuMemoryOrder == CE_ACQ ? "acquire" : gpuMemoryOrder == CE_REL ? "release" : gpuMemoryOrder == CE_ACQ_ACQ ? "acq/acq" : gpuMemoryOrder == CE_ACQ_REL ? "acq/rel" : "non-atomic", outerLoopCount == CE_BASE ? "1" : outerLoopCount == CE_1K ? "1K" : outerLoopCount == CE_10K ? "10K" : outerLoopCount == CE_100K ? "100K" : outerLoopCount == CE_1M ? "1M" : outerLoopCount == CE_10M ? "10M" : outerLoopCount == CE_100M ? "100M" : outerLoopCount == CE_1B ? "1B" : "unknown");


    //randomly pad all the objects
    if (objectType != CE_LOADED) {
        if (memoryType == CE_GDDR) {
            for (int i = 0; i < (*count); i++) {
                initRandString((sizeof(LargeLinkedObject) - sizeof(int)) / (2 * sizeof(char)));

                strcpy(localCopy[localOrder[i]].padding1, randString);
                initRandString((sizeof(LargeLinkedObject) - sizeof(int)) / (2 * sizeof(char)));
                strcpy(localCopy[localOrder[i]].padding2, randString);

                localCopy[localOrder[i]].data_na = localOrder[(i + 1) % *count];
                localCopy[localOrder[i]].data.store(localOrder[(i + 1) % *count]);
            }

            SAFE(cudaMemcpy(largeObjectList, localCopy, sizeof(struct LargeLinkedObject) * *count, cudaMemcpyHostToDevice));
        } else {
            for (int i = 0; i < (*count); i++) {
                initRandString((sizeof(LargeLinkedObject) - sizeof(int)) / (2 * sizeof(char)));

                strcpy(largeObjectList[localOrder[i]].padding1, randString);
                initRandString((sizeof(LargeLinkedObject) - sizeof(int)) / (2 * sizeof(char)));
                strcpy(largeObjectList[localOrder[i]].padding2, randString);

                largeObjectList[localOrder[i]].data_na = localOrder[(i + 1) % *count];
                largeObjectList[localOrder[i]].data.store(localOrder[(i + 1) % *count]);
            }
        }
    } else {
        if (memoryType == CE_GDDR) {
            struct LargeLinkedObject * localCopy = (struct LargeLinkedObject *) malloc(sizeof(struct LargeLinkedObject) * *count);

            for (int i = 0; i < *count; i++) {
                // for (int z = 0; z < 2; z++) {
                //     for (int j = 0; j < PADDING_LENGTH / 4; j++) {
                //         // localCopy[i].data_na_list[j] = i * z * j;
                //         // localCopy[i].data_list[j].store(i * z * j);
                //     }
                // }
                localCopy[i].data_na = localOrder[(i + 1) % *count];
                localCopy[i].data.store(localOrder[(i + 1) % *count]);
            }

            SAFE(cudaMemcpy(largeObjectList, localCopy, sizeof(struct LargeLinkedObject) * *count, cudaMemcpyHostToDevice));
        } else {
            for (int i = 0; i < *count; i++) {
                // for (int z = 0; z < 2; z++) {
                //     for (int j = 0; j < PADDING_LENGTH / 4; j++) {
                //         largeObjectList[i].data_na_list[j] = i * z * j;
                //         largeObjectList[i].data_list[j].store(i * z * j);
                //     }
                // }
                largeObjectList[i].data_na = localOrder[(i + 1) % *count];
                largeObjectList[i].data.store(localOrder[(i + 1) % *count]);
            }
        }
    }
    
    printSequence(operationSequence);

    int CPUEventCount = 0;
    int GPUEventCount = 0;

    if (warmup && GPUConsumeFirst == 1) {
        printf("Warming up GPU\n");
        switch (gpuMemoryOrder) {
            case CE_ACQ:
                if (objectType == CE_ARRAY){
                    GPUListConsumer_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);}
                else
                    GPULinkedListConsumer_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_REL:
                if (objectType == CE_ARRAY)
                    GPUListConsumer_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                else 
                    GPULinkedListConsumer_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_ACQ_ACQ:
                if (objectType == CE_ARRAY)
                    GPUListConsumer_acq_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                else
                    GPULinkedListConsumer_acq_acq<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_ACQ_REL:
                if (objectType == CE_ARRAY)
                    GPUListConsumer_acq_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                else
                    GPULinkedListConsumer_acq_rel<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
            case CE_NONE:
                if (objectType == CE_ARRAY)
                    GPUListConsumer<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                else
                    GPULinkedListConsumer<<<1,1>>>(flag, largeObjectList, largeObjectList, largeObjectListConsumer, count, beforeLoop[GPUEventCount], afterLoop[GPUEventCount]);
                break;
        }
        cudaDeviceSynchronize();
    } else {
        printf("Skipping GPU Warmup\n");
    }

    // do all the operations in the order specified
    for (int i = 0; i < operationSequence[0].total; i++) {
        if (operationSequence[i].device == CE_CPU) {
            begin[CPUEventCount] = std::chrono::high_resolution_clock::now();

            callFunction(
                {
                    operationSequence[i].device, 
                    operationSequence[i].action, 
                    cpuMemoryOrder, 
                    outerLoopCount, 
                    objectType,
                    memoryType,
                },
                flag, 
                largeObjectList,
                localCopy,
                largeObjectListConsumer, 
                localConsumer,
                loadedListConsumer, 
                localLoadedConsumer,
                largeObjectListOrder, 
                localOrder,
                count, 
                beforeLoop[GPUEventCount], 
                afterLoop[GPUEventCount]);
            
            end[CPUEventCount++] = std::chrono::high_resolution_clock::now();
        } else {
            cudaEventRecord(start[GPUEventCount]);
            
            callFunction(
                {
                    operationSequence[i].device, 
                    operationSequence[i].action, 
                    gpuMemoryOrder, 
                    outerLoopCount, 
                    objectType,
                    memoryType,
                },
                flag, 
                largeObjectList,
                localCopy,
                largeObjectListConsumer, 
                localConsumer,
                loadedListConsumer, 
                localLoadedConsumer,
                largeObjectListOrder, 
                localOrder,
                count, 
                beforeLoop[GPUEventCount], 
                afterLoop[GPUEventCount]);

            cudaEventRecord(stop[GPUEventCount]);

            // synchronize GPU executation after every operation
            cudaEventSynchronize(stop[GPUEventCount]);
            GPUEventCount++;
        }
    }

    cudaDeviceSynchronize();

    // SAFE(cudaMemcpy(localBeforeLoop, beforeLoop, sizeof(unsigned int *) * numGPUEvents, cudaMemcpyDeviceToHost));
    // SAFE(cudaMemcpy(localAfterLoop, afterLoop, sizeof(unsigned int *) * numGPUEvents, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numGPUEvents; i++) {
        milliseconds[i] = 0;
        cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
        milliseconds[i] *= 1e6;

        if (objectType == CE_LOADED) {
            milliseconds[i] /= PADDING_LENGTH / 4;
        }

        loopDuration[i] = localAfterLoop[i] - localBeforeLoop[i];
    }

    for (int i = 0; i < numCPUEvents; i++) {
        durations[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end[i] - begin[i]).count();
    }

    printResults(operationSequence, numCPUEvents, numGPUEvents, durations, loopDuration, milliseconds, count);

    printf("\n----------------------\n\n\n");

    return 0;
}
