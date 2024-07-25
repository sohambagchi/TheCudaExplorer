#ifndef THECUDAEXPLORER_LOADED_H_
#define THECUDAEXPLORER_LOADED_H_

#include "theCudaExplorer_top.cuh"

template <typename T>
__global__ void GPULoadedListConsumer(cuda::atomic<int>* flag, T* ptr, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr[order[i]].data_na_list[j];
      }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPULoadedListConsumer_1K(cuda::atomic<int>* flag, T* ptr, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int z = 0; z < 1000; z++) {
      for (int i = 0; i < _count; i++) {
        for (int j = 0; j < PADDING_LENGTH; j++) {
          result[i][j] = ptr[order[i]].data_na_list[j];
        }
      }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPULoadedListConsumer_10K(cuda::atomic<int>* flag, T* ptr, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int z = 0; z < 10000; z++) {
      for (int i = 0; i < _count; i++) {
        for (int j = 0; j < PADDING_LENGTH; j++) {
          result[i][j] = ptr[order[i]].data_na_list[j];
        }
      }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPULoadedListConsumer_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 1000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 10000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_relaxed);
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_rel_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 1000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_rel_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 10000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_relaxed);
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_rel_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 1000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
        result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_rel_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 10000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
        result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_acquire);
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_acq_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 1000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
        result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_acquire);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListConsumer_acq_acq_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int z = 0; z < 10000; z++) {
    for (int i = 0; i < _count; i++) {
      for (int j = 0; j < PADDING_LENGTH; j++) {
        result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
        result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_acquire);
      }
    }
  }
}

template <typename T>
__global__ void GPULoadedListProducer(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif

  int _count = *count;
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      ptr[order[i]].data_na_list[j] = i;
    }
  }
}

template <typename T>
__global__ void GPULoadedListProducer_rel(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
  #ifdef RC
  flag->store(0, cuda::memory_order_release);
  #endif
  int _count = *count;
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      ptr[order[i]].data_list[j].store(i, cuda::memory_order_relaxed);
    }
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPULoadedListConsumer(cuda::atomic<int>* flag, T* ptr, int **result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr[order[i]].data_na_list[j];
    }
  }
}

template <typename T>
__host__ void CPULoadedListConsumer_acq(cuda::atomic<int>* flag, T* ptr1, int **result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
    }
  }
}

template <typename T>
__host__ void CPULoadedListConsumer_rel(cuda::atomic<int>* flag, T* ptr1, int **result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_relaxed);
    }
  }
}

template <typename T>
__host__ void CPULoadedListConsumer_acq_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_relaxed);
    }
  }
}

template <typename T>
__host__ void CPULoadedListConsumer_acq_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int **result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      result[i][j] = ptr1[order[i]].data_list[j].load(cuda::memory_order_acquire);
      result[i][j] = ptr2[order[i]].data_list[j].load(cuda::memory_order_acquire);
    }
  }
}

template <typename T>
__host__ void CPULoadedListProducer(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
  #ifdef RC
  flag->store(0, cuda::memory_order_release);
  #endif
  int _count = *count;
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      ptr[order[i]].data_na_list[j] = i;
    }
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPULoadedListProducer_rel(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
  #ifdef RC
  flag->store(0, cuda::memory_order_release);
  #endif
  int _count = *count;
  for (int i = 0; i < _count; i++) {
    for (int j = 0; j < PADDING_LENGTH; j++) {
      ptr[order[i]].data_list[j].store(i, cuda::memory_order_relaxed);
    }
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

#endif