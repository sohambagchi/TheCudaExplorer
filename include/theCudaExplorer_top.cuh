#ifndef THECUDAEXPLORER_TOP_H_
#define THECUDAEXPLORER_TOP_H_

#include <cuda/atomic>
#include "theCudaExplorer_array.cuh"
#include "theCudaExplorer_list.cuh"
#include "theCudaExplorer_loaded.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>

// Uncomment line below to enable release/acquire flags
// #define RC

#ifndef PADDING_LENGTH
#define PADDING_LENGTH 4
#endif

#ifndef SCOPE
#define SCOPE cuda::thread_scope_system
#endif

typedef enum {
    CE_CPU,
    CE_GPU
} CEDevice;

typedef enum {
    CE_LOAD,
    CE_STORE
} CEAction;

typedef enum {
    CE_DRAM,
    CE_UM,
    CE_GDDR,
    CE_SYS
} CEMemory;

typedef enum {
    CE_NONE,
    CE_ACQ,
    CE_REL,
    CE_ACQ_ACQ,
    CE_ACQ_REL,
} CEOrder;

typedef enum {
    CE_BASE,
    CE_1K,
    CE_10K,
    CE_100K,
    CE_1M,
    CE_10M,
    CE_100M,
    CE_1B,
} CECount;

typedef enum {
    CE_ARRAY,
    CE_LINKEDLIST,
    CE_LOADED,
} CEObjectType;

typedef struct {
    CEDevice device;
    CEAction action;
    int total;
} CEOperation;

struct LargeObject {
    char padding1[PADDING_LENGTH];
    // int data;
    int data_na;
    char padding2[PADDING_LENGTH];
    cuda::atomic<int, SCOPE> data;
};

struct LargeLinkedObject {
    char padding1[PADDING_LENGTH];
    int data_na;
    char padding2[PADDING_LENGTH];
    cuda::atomic<int, SCOPE> data;
    // struct LargeLinkedObject *next;
    // int next;
    int data_na_list[0];
    cuda::atomic<int, SCOPE> data_list[0];
};

struct LoadedLargeObject {
    int data_na_list[PADDING_LENGTH/4];
    int data_na;
    cuda::atomic<int, SCOPE> data_list[PADDING_LENGTH/4];
    cuda::atomic<int, SCOPE> data;
}; 

__global__ void EmptyKernel(int *count, unsigned int *before, unsigned int *after) {
  // *before = clock64();
  // for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 128; j++) {
      asm volatile("");
    }
  // }
  // *after = clock64();
  // return;
}

template <typename T>
void CEMemcpyDToH(T *dst, T *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

template <typename T>
void CEMemcpyDToH_1K(T *dst, T *src, size_t size) {
    for (int z = 0; z < 1000; z++) 
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

template <typename T>
void CEMemcpyHToD(T *dst, T *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

template <typename T>
void CEMemcpyHToD_1K(T *dst, T *src, size_t size) {
    for (int z = 0; z < 1000; z++) 
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

template <typename T>
void callFunction(const std::vector<int>& params, cuda::atomic<int>* flag, T* ptr, T* localPtr, int *result, int *localResult, int *loadedResult, int *localLoadedResult, int *order, int *localOrder, int *count, unsigned int *before, unsigned int *after) {
    CEDevice ceDevice = (CEDevice) params[0];
    CEAction ceAction = (CEAction) params[1];
    CEOrder ceOrder = (CEOrder) params[2];
    CECount ceCount = (CECount) params[3];
    CEObjectType ceObjectType = (CEObjectType) params[4];
    CEMemory ceMemory = (CEMemory) params[5];

    switch (ceDevice) {
        case CE_CPU:
            switch (ceAction) {
                case CE_LOAD:
                    switch (ceMemory) {
                        case CE_GDDR:
                            switch (ceCount) {
                                case CE_BASE:
                                    CEMemcpyDToH(localPtr, ptr, *count * sizeof(T));
                                    break;
                                case CE_1K:
                                    CEMemcpyDToH_1K(localPtr, ptr, *count * sizeof(T));
                                    break;
                            }
                            break;
                        default:
                            switch (ceOrder) {
                                case CE_NONE:
                                    switch (ceCount) {
                                        case CE_BASE:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer(flag, ptr, loadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        case CE_1K:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_1K(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_1K(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_1K(flag, ptr, loadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                    }
                                    break;
                                case CE_ACQ:
                                    switch (ceCount) {
                                        case CE_BASE:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_acq(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_acq(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_acq(flag, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        case CE_1K:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_acq_1K(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_acq_1K(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_acq_1K(flag, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                    }
                                    break;
                                case CE_REL:
                                    switch (ceCount) {
                                        case CE_BASE:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_rel(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_rel(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_rel(flag, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        case CE_1K:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_rel_1K(flag, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_rel_1K(flag, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_rel_1K(flag, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                    }
                                    break;
                                case CE_ACQ_ACQ:
                                    switch (ceCount) {
                                        default:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_acq_acq(flag, ptr, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_acq_acq(flag, ptr, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_acq_acq(flag, ptr, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        // case CE_1K:
                                        //     switch (ceObjectType) {
                                        //         case CE_ARRAY:
                                        //             CPUListConsumer_acq_acq_1K(flag, ptr, ptr, localResult, localOrder, count);
                                        //             break;
                                        //         case CE_LINKEDLIST:
                                        //             CPULinkedListConsumer_acq_acq_1K(flag, ptr, ptr, localResult, count);
                                        //             break;
                                        //         case CE_LOADED:
                                        //             CPULoadedListConsumer_acq_acq_1K(flag, ptr, ptr, localLoadedResult, localOrder, count);
                                        //             break;
                                        //     }
                                        //     break;
                                    }
                                    break;
                                case CE_ACQ_REL:
                                    switch (ceCount) {
                                        default:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListConsumer_acq_rel(flag, ptr, ptr, localResult, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListConsumer_acq_rel(flag, ptr, ptr, localResult, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListConsumer_acq_rel(flag, ptr, ptr, localLoadedResult, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        // case CE_1K:
                                        //     switch (ceObjectType) {
                                        //         case CE_ARRAY:
                                        //             CPUListConsumer_acq_rel_1K(flag, ptr, ptr, localResult, localOrder, count);
                                        //             break;
                                        //         case CE_LINKEDLIST:
                                        //             CPULinkedListConsumer_acq_rel_1K(flag, ptr, ptr, localResult, count);
                                        //             break;
                                        //         case CE_LOADED:
                                        //             CPULoadedListConsumer_acq_rel_1K(flag, ptr, ptr, localLoadedResult, localOrder, count);
                                        //             break;
                                        //     }
                                        //     break;
                                    }
                                    break;
                            }
                            break;
                    }
                    break;
                case CE_STORE:
                    switch (ceMemory) {
                        case CE_GDDR:
                            switch (ceCount) {
                                case CE_BASE:
                                    CEMemcpyHToD(ptr, localPtr, *count * sizeof(T));
                                    break;
                                case CE_1K:
                                    CEMemcpyHToD_1K(ptr, localPtr, *count * sizeof(T));
                                    break;
                            }
                            break;
                        default:
                            switch (ceOrder) {
                                case CE_NONE:
                                    switch (ceCount) {
                                        default:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListProducer(flag, ptr, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListProducer(flag, ptr, localOrder, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListProducer(flag, ptr, order, count);
                                                    break;
                                            }
                                            break;
                                        // case CE_1K:
                                        //     switch (ceObjectType) {
                                        //         case CE_ARRAY:
                                        //             CPUListProducer_1K(flag, ptr, localOrder, count);
                                        //             break;
                                        //         case CE_LINKEDLIST:
                                        //             CPULinkedListProducer_1K(flag, ptr, localOrder, count);
                                        //             break;
                                        //         case CE_LOADED:
                                        //             CPULoadedListProducer_1K(flag, ptr, order, count);
                                        //             break;
                                        //     }
                                        //     break;
                                    }
                                    break;
                                default:
                                    switch (ceCount) {
                                        default:
                                            switch (ceObjectType) {
                                                case CE_ARRAY:
                                                    CPUListProducer_rel(flag, ptr, localOrder, count);
                                                    break;
                                                case CE_LINKEDLIST:
                                                    CPULinkedListProducer_rel(flag, ptr, localOrder, count);
                                                    break;
                                                case CE_LOADED:
                                                    CPULoadedListProducer_rel(flag, ptr, localOrder, count);
                                                    break;
                                            }
                                            break;
                                        // case CE_1K:
                                        //     switch (ceObjectType) {
                                        //         case CE_ARRAY:
                                        //             CPUListProducer_rel_1K(flag, ptr, localOrder, count);
                                        //             break;
                                        //         case CE_LINKEDLIST:
                                        //             CPULinkedListProducer_rel_1K(flag, ptr, localOrder, count);
                                        //             break;
                                        //         case CE_LOADED:
                                        //             CPULoadedListProducer_rel_1K(flag, ptr, localOrder, count);
                                        //             break;
                                        //     }
                                        //     break;
                                    }
                                    break;
                            }
                            break;
                    }
                    break;
            }
            break;
        case CE_GPU:
            switch (ceAction) {
                case CE_LOAD:
                    switch (ceOrder) {
                        case CE_NONE:
                            switch (ceCount) {
                                case CE_BASE:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer<<<1,1>>>(flag, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_1K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_1K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_1K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_1K<<<1,1>>>(flag, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_10K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_10K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_10K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_10K<<<1,1>>>(flag, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                            }
                            break;
                        case CE_ACQ:
                            switch (ceCount) {
                                case CE_BASE:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_1K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_1K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_1K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_1K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_10K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_10K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_10K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_10K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                            }
                            break;
                        case CE_REL:
                            switch (ceCount) {
                                case CE_BASE:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_rel<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_rel<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_rel<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_1K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_rel_1K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_rel_1K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_rel_1K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_10K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_rel_10K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_rel_10K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_rel_10K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                            }
                            break;
                        case CE_ACQ_ACQ:
                            switch (ceCount) {
                                case CE_BASE:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_acq<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_acq<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_acq<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_1K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_acq_1K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_acq_1K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_acq_1K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_10K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_acq_10K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_acq_10K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_acq_10K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                            }
                            break;
                        case CE_ACQ_REL:
                            switch (ceCount) {
                                case CE_BASE:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_rel<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_rel<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_rel<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_1K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_rel_1K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_rel_1K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_rel_1K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                                case CE_10K:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListConsumer_acq_rel_10K<<<1,1>>>(flag, ptr, ptr, result, order, count, before, after);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListConsumer_acq_rel_10K<<<1,1>>>(flag, ptr, ptr, result, count, before, after);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListConsumer_acq_rel_10K<<<1,1>>>(flag, ptr, ptr, loadedResult, order, count, before, after);
                                            break;
                                    }
                                    break;
                            }
                            break;
                    }
                    break;
                case CE_STORE:
                    switch (ceOrder) {
                        case CE_NONE:
                            switch (ceCount) {
                                default:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListProducer<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListProducer<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListProducer<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                    }
                                    break;
                            }
                            break;
                        default:
                            switch (ceCount) {
                                default:
                                    switch (ceObjectType) {
                                        case CE_ARRAY:
                                            GPUListProducer_rel<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                        case CE_LINKEDLIST:
                                            GPULinkedListProducer_rel<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                        case CE_LOADED:
                                            GPULoadedListProducer_rel<<<1,1>>>(flag, ptr, order, count);
                                            break;
                                    }
                                    break;
                            }
                            break;
                    }
                    break;
            }
            break;
    }
}

#endif