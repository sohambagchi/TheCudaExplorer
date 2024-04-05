#include <cstdio>
#include <chrono>

#include "theCudaExplorer.cuh"

char randString[3556156];

// creates a random string for the object padding
void initRandString(int paddingSize) {
    for (int i = 0; i < paddingSize; i++) {
        randString[i] = rand() % 26 + 'a';
    }
}

// creates the shuffling order for the objects
void shuffleList(int * localOrder, int count) {
    int j, tmp;
    for (int i = 0; i < count; i++) {
        j = rand() % count;
        tmp = localOrder[i];
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

    printf("\n\n\n");

}

void printResults(CEOperation * sequence, int numCPUEvents, int numGPUEvents, int64_t * durations, float * milliseconds, int * count) {
    
    printf("Results\n======================\n");

    for (int i = 0, j = 0, k = 0; i < sequence[0].total; i++) {
        printf("%s %s\n", sequence[i].device == CE_CPU ? "CPU" : "\t\tGPU", sequence[i].action == CE_LOAD ? "ld" : "st");
        printf("%s    (%ld ns)\n", sequence[i].device == CE_CPU ? "" : "\t\t", sequence[i].device == CE_CPU ? durations[j++] / *count : (int64_t) (milliseconds[k++] / *count));
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

    printf("Size of Object: %.2f MB\n", sizeof(struct LargeObject) / (1024.0 * 1024.0));
    printf("Number of Objects: %d\n", numObjects);
    printf("CPU Events Timed: %d\t GPU Events Timed: %d\n\n", numCPUEvents, numGPUEvents);


    // CPU Timers
    std::chrono::high_resolution_clock::time_point begin[numCPUEvents], end[numCPUEvents];
    int64_t durations[numCPUEvents];

    // CUDA Timers
    cudaEvent_t start[numGPUEvents], stop[numGPUEvents];
    float milliseconds[numGPUEvents];

    for (int i = 0; i < numGPUEvents; i++) {
        SAFE(cudaEventCreate(&start[i]));
        SAFE(cudaEventCreate(&stop[i]));
    }

    printf("- Timers Initialized\n");

    cuda::atomic<int> * flag;
    struct LargeObject * largeObjectList;
    int * largeObjectListConsumer;
    int * localConsumer;
    int * largeObjectListOrder;
    int * localOrder;
    int * count;

    long long int * before;
    long long int * after;
    long long int * localBefore;
    long long int * localAfter;

    SAFE(cudaMallocHost(&flag, sizeof(cuda::atomic<int>)));
    SAFE(cudaMallocHost(&count, sizeof(int)));
    SAFE(cudaMallocHost(&localConsumer, sizeof(int*) * numObjects));
    SAFE(cudaMallocHost(&localOrder, sizeof(int*) * numObjects));
    
    printf("- CPU Objects Allocated\n");

    SAFE(cudaMalloc(&largeObjectListConsumer, sizeof(int*) * numObjects));
    SAFE(cudaMalloc(&largeObjectListOrder, sizeof(int*) * numObjects));
    SAFE(cudaMalloc(&before, sizeof(long long int) * numGPUEvents));
    SAFE(cudaMalloc(&after, sizeof(long long int) * numGPUEvents));
    SAFE(cudaMallocHost(&localBefore, sizeof(long long int) * numGPUEvents));
    SAFE(cudaMallocHost(&localAfter, sizeof(long long int) * numGPUEvents));

    printf("- GPU Objects Allocated\n");

    if (memoryType == CE_DRAM) {
        SAFE(cudaMallocHost(&largeObjectList, sizeof(struct LargeObject) * numObjects));
    } else if (memoryType == CE_UM) {
        SAFE(cudaMallocManaged(&largeObjectList, sizeof(struct LargeObject) * numObjects));
    } 

    printf("- Large Objects Allocated\n");


    *count = numObjects;

    // allocate the ordering array in both DRAM and GDDR
    for (int i = 0; i < (*count); i++) {
        localOrder[i] = i;
    }

    shuffleList(localOrder, *count);

    // for (int i = 0; i < (*count); i++) {

    //     // Allocate the data objects according to arguments
    //     // if (memoryType == CE_DRAM) {
    //     //     SAFE(cudaMallocHost(&largeObjectList[*localOrder[i]], sizeof(struct LargeObject)));
    //     // } else {
    //     //     SAFE(cudaMallocManaged(&largeObjectList[*localOrder[i]], sizeof(struct LargeObject)));
    //     // }

    //     // Separate Consumer Lists for CPU and GPU, to mitigate remote store latency
    //     // SAFE(cudaMalloc(&largeObjectListConsumer[i], sizeof(int)));
    //     // SAFE(cudaMallocHost(&localConsumer[i], sizeof(int)));

    //     // Copy the locally shuffled order to the device
    //     // SAFE(cudaMemcpy(largeObjectListOrder[i], localOrder[i], sizeof(int), cudaMemcpyHostToDevice));
    // }



    printf("\nUsing %s for Objects\n\n", memoryType == CE_DRAM ? "cudaMallocHost" : "cudaMallocManaged");

    printSequence(operationSequence);

    // randomly pad all the objects
    for (int i = 0; i < (*count); i++) {
        initRandString((sizeof(LargeObject) - sizeof(int)) / (2 * sizeof(char)));
        strcpy((largeObjectList[localOrder[i]]).padding1, randString);
        (largeObjectList[localOrder[i]]).data = i;
        strcpy((largeObjectList[localOrder[i]]).padding2, randString);
    }

    int CPUEventCount = 0;
    int GPUEventCount = 0;

    // do all the operations in the order specified
    for (int i = 0; i < operationSequence[0].total; i++) {
        if (operationSequence[i].device == CE_CPU) {
            begin[CPUEventCount] = std::chrono::high_resolution_clock::now();
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    if (numObjects == 1) 
                        CPUSingleConsumer(flag, largeObjectList, localConsumer);
                    // else 
                        // CPUListConsumer_(flag, largeObjectList, localConsumer, localOrder, count);
                    break;
                case CE_STORE:
                    if (numObjects == 1) 
                        CPUSingleProducer(flag, largeObjectList, rand()%100);
                    // else
                        // CPUListProducer_(flag, largeObjectList, localOrder, count);
                    break;
            }
            end[CPUEventCount++] = std::chrono::high_resolution_clock::now();
        } else {
            cudaEventRecord(start[GPUEventCount]);
            switch (operationSequence[i].action) {
                case CE_LOAD:
                    if (numObjects == 1) 
                        GPUSingleConsumer<<<1,1>>>(flag, largeObjectList, largeObjectListConsumer, &before[GPUEventCount], &after[GPUEventCount]);
                    // else
                    //     GPUListConsumer_<<<1,1>>>(flag, largeObjectList, largeObjectListConsumer, largeObjectListOrder, count);
                    break;
                case CE_STORE:
                    if (numObjects == 1) 
                        GPUSingleProducer<<<1,1>>>(flag, largeObjectList, &before[GPUEventCount], &after[GPUEventCount], rand()%100);
                    // else
                        // GPUListProducer_<<<1,1>>>(flag, largeObjectList, largeObjectListOrder, count);
                    break;
            }
            cudaEventRecord(stop[GPUEventCount]);

            // synchronize GPU executation after every operation
            cudaEventSynchronize(stop[GPUEventCount]);
            GPUEventCount++;
        }
    }

    // struct LargeObject * GPUObjectList;
    // SAFE(cudaMalloc(&GPUObjectList, sizeof(struct LargeObject) * numObjects));
    // SAFE(cudaMemcpy(GPUObjectList, largeObjectList, sizeof(struct LargeObject) * numObjects, cudaMemcpyHostToDevice));

    // printf("succ");

    cudaDeviceSynchronize();

    for (int i = 0; i < numGPUEvents; i++) {
        milliseconds[i] = 0;
        cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
        milliseconds[i] *= 1e6;
    }

    for (int i = 0; i < numCPUEvents; i++) {
        durations[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end[i] - begin[i]).count();
    }

    printResults(operationSequence, numCPUEvents, numGPUEvents, durations, milliseconds, count);

    cudaMemcpy(localBefore, before, sizeof(long long int) * numGPUEvents, cudaMemcpyDeviceToHost);
    cudaMemcpy(localAfter, after, sizeof(long long int) * numGPUEvents, cudaMemcpyDeviceToHost);

    printf("Clock64 Results\n");

    for (int i = 0; i < numGPUEvents; i++) {
        printf("GPU Event %d: %lld cycles\n", i, localAfter[i] - localBefore[i]);
    }

    // printf("%d\n", largeObjectList[0].data);

    cudaMemcpy(localConsumer, largeObjectListConsumer, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("%d\n", localConsumer[0]);

    // printf("%d\n", largeObjectList[0].data);

    return 0;


    // LABEL:MH40
    // struct LargeObject ** remoteObjectList;
    
    // SAFE(cudaMallocHost(&remoteObjectList, sizeof(struct LargeObject*) * *count));
      
    // for (int i = 0; i < (*count); i++) {
    //     SAFE(cudaMalloc(&remoteObjectList[*localOrder[i]], sizeof(struct LargeObject)));
    // }

    // SAFE(cudaMemcpy(remoteObjectList[*localOrder[i]], largeObjectList[*localOrder[i]], sizeof(struct LargeObject), cudaMemcpyHostToDevice));

}