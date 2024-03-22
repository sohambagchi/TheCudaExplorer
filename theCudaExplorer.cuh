#include <cuda/atomic>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Uncomment line below to enable release/acquire flags
// #define RC

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
    CE_UM
} CEMemory;

typedef struct {
    CEDevice device;
    CEAction action;
    int total;
} CEOperation;

struct LargeObject {
    char padding1[3556156];
    int data;
    char padding2[3556156];
};

struct LargeLinkedObject {
    char padding1[3556156];
    int data;
    char padding2[3556156];
    struct LargeLinkedObject *next;
};

template <typename T>
__global__ void GPUListConsumer(cuda::atomic<int>* flag, T** ptr, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    *result[i] = (ptr[*order[i]])->data;
  }
}

template <typename T>
__host__ void CPUListConsumer(cuda::atomic<int>* flag, T** ptr, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    *result[i] = (ptr[*order[i]])->data;
  }
}

template <typename T>
__global__ void GPUListProducer(cuda::atomic<int>* flag, T** ptr, int ** order, int *count) {
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif

  for (int i = 0; i < *count; i++) {
    (ptr[*order[i]])->data = i;
  }

    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

template <typename T>
__host__ void CPUListProducer(cuda::atomic<int>* flag, T** ptr, int ** order, int *count) {
    
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif
  
  for (int i = 0; i < *count; i++) {
    (ptr[*order[i]])->data = i;
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