#include <cuda/atomic>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Uncomment line below to enable release/acquire flags
// #define RC
#define PADDING_LENGTH 33554432 >> 1

// #define SCOPE cuda::thread_scope_thread
// #define LOAD1 cuda::memory_order_acquire
// #define LOAD2 cuda::memory_order_relaxed

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
    CE_GDDR
} CEMemory;

typedef struct {
    CEDevice device;
    CEAction action;
    int total;
} CEOperation;

struct LargeObject {
    char padding1[PADDING_LENGTH];
    // int data;
    cuda::atomic<int, SCOPE> data;
    char padding2[PADDING_LENGTH];
};

struct LargeLinkedObject {
    char padding1[PADDING_LENGTH];
    int data;
    char padding2[PADDING_LENGTH];
    struct LargeLinkedObject *next;
};

// Loads from the array of objects in the shuffled order into the result array (which is in GDDR)
template <typename T>
__global__ void GPUListConsumer(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int * result, int * order, int *count, unsigned int * before, unsigned int * after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  // int x;
  // int z = *count;

  // int x[512];
  // int y[512];
  // int z[512];

  // *before = clock64();

  int _count = *count;

  for (int i = 0; i < _count; i++) {
    // result[i] = (ptr[order[i]]).data;
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr1[order[i]]).data.load(LOAD1);
    
    
    // result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
    result[i] = (ptr2[order[i]]).data.load(LOAD2);
    // x[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    // result[i] = x[i];
    // y[i] = (ptr[order[i]]).data.load(cuda::memory_order_acquire);
    // y[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
    // result[i] += y[i];
    // x = (ptr[order[i]]).data;
    // z[i] = x[i] + y[i];
  }

  // *after = clock64();

  // for (int i = 0; i < *count; i++) {
  //   result[i] = x[i] + y[i];
  // }
}

// Stores to the array of objects in the shuffled order
template <typename T>
__global__ void GPUListProducer(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif

  for (int i = 0; i < *count; i++) {
    // (ptr[order[i]]).data = i;
    (ptr[order[i]]).data.store(i, cuda::memory_order_release);
  }

    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

// Loads from the array of objects in the shuffled order into the result array (which is in DRAM)
template <typename T>
__host__ void CPUListConsumer(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr[order[i]]).data.load(cuda::memory_order_relaxed);
  }
}

// Stores to the array of objects in the shuffled order
template <typename T>
__host__ void CPUListProducer(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif
  
  for (int i = 0; i < *count; i++) {
    // (ptr[order[i]]).data = i;
    (ptr[order[i]]).data.store(i, cuda::memory_order_relaxed);
  }
  
    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

template <typename T>
__global__ void GPULinkedListConsumer(cuda::atomic<int>* flag, T* head, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head;
  for (int i = 0; i < *count; i++) {
    *result[i] = current->data;
    current = current->next;
  }
}

template <typename T>
__host__ void CPULinkedListConsumer(cuda::atomic<int>* flag, T* head, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head;
  for (int i = 0; i < *count; i++) {
    *result[i] = current->data;
    current = current->next;
  }
}

template <typename T>
__global__ void GPULinkedListProducer(cuda::atomic<int>* flag, T* head, int ** order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current->data = i;
    current = current->next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPULinkedListProducer(cuda::atomic<int>* flag, T* head, int ** order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current->data = i;
    current = current->next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}