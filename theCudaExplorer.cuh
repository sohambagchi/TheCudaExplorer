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
    CE_UM,
    CE_GDDR,
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
    int data2;
};

struct LargeLinkedObject {
    char padding1[3556156];
    int data;
    char padding2[3556156];
    struct LargeLinkedObject *next;
};

// Loads from the array of objects in the shuffled order into the result array (which is in GDDR)
template <typename T>
__global__ void GPUListConsumer_(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    result[i] = (ptr[order[i]]).data;
  }
}

// Loads from the array of objects in the shuffled order into the result array (which is in DRAM)
template <typename T>
__host__ void CPUListConsumer_(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    result[i] = (ptr[order[i]]).data;
  }
}

// Stores to the array of objects in the shuffled order
template <typename T>
__global__ void GPUListProducer_(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif

  for (int i = 0; i < *count; i++) {
    (ptr[order[i]]).data = i;
  }

    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

// Stores to the array of objects in the shuffled order
template <typename T>
__host__ void CPUListProducer_(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif
  
  for (int i = 0; i < *count; i++) {
    (ptr[order[i]]).data = i;
  }
  
    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}


// Loads from the array of objects in the shuffled order into the result array (which is in GDDR)
template <typename T>
__global__ void GPUListConsumer(cuda::atomic<int>* flag, T** ptr, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    *result[i] = (ptr[*order[i]])->data;
  }
}

// Loads from the array of objects in the shuffled order into the result array (which is in DRAM)
template <typename T>
__host__ void CPUListConsumer(cuda::atomic<int>* flag, T** ptr, int ** result, int ** order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    *result[i] = (ptr[*order[i]])->data;
  }
}

// Stores to the array of objects in the shuffled order
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

// Stores to the array of objects in the shuffled order
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

template <typename T>
__global__ void GPUSingleConsumer(cuda::atomic<int>* flag, T* ptr, int * result, long long * before, long long * after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  int x, y;

  long long int a, b;

  asm volatile("membar.sys;");

  asm volatile("mov.u64 %0, %clock64;" : "=l"(a));
  
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(&ptr->data));
  // asm volatile("ld.global.u32 %0, [%1];" : "=r"(y) : "l"(&ptr->data2));
  // asm("ld.global.u32 %0, [%1];" : "=r"(y) : "l"(ptr));
  asm volatile("membar.gl;");
  asm volatile("mov.u64 %0, %clock64;" : "=l"(b));

  // *before = clock64();
  // *result = ptr->data;
  // *after = clock64();
  *result = x;
  // *result = y;

  *before = a;
  *after = b;
}

template <typename T>
__global__ void GPUSingleConsumer_(cuda::atomic<int>* flag, T* ptr, int * result, long long * before, long long * after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  int x, y;

  long long int a, b;

  asm("mov.u64 %0, %clock64;" : "=l"(a));
  asm("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(ptr));
  asm("ld.global.u32 %0, [%1];" : "=r"(y) : "l"(ptr));
  asm("mov.u64 %0, %clock64;" : "=l"(b));

  // *before = clock64();
  // *result = ptr->data;
  // *after = clock64();
  *result = x;
  *result = y;

  *before = a;
  *after = b;
}

template <typename T>
__host__ void CPUSingleConsumer(cuda::atomic<int>* flag, T* ptr, int * result) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  *result = ptr->data;

}

template <typename T>
__global__ void GPUSingleProducer(cuda::atomic<int>* flag, T* ptr, long long * before, long long * after, int rand) {
  *before = clock64();
  ptr->data = 69;
  *after = clock64();
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPUSingleProducer(cuda::atomic<int>* flag, T* ptr, int rand) {
  ptr->data = 42*rand;
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}