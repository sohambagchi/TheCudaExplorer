#include <cuda/atomic>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Uncomment line below to enable release/acquire flags
// #define RC
// #define PADDING_LENGTH 33554432 >> 1

// #define SCOPE cuda::thread_scope_thread
// #define LOAD1 cuda::memory_order_acquire
// #define LOAD2 cuda::memory_order_relaxed

// #define TWO_LOADS

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

typedef enum {
    CE_ACQ,
    CE_REL,
    CE_ACQ_ACQ,
    CE_ACQ_REL,
    CE_NONE,
} CEOrder;

typedef enum {
    CE_1K,
    CE_10K,
    CE_100K,
    CE_1M,
    CE_10M,
    CE_100M,
    CE_1B,
    CE_BASE,
} CECount;

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
    int data;
    char padding2[PADDING_LENGTH];
    struct LargeLinkedObject *next;
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
__global__ void GPUListConsumer_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_100K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_1M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_10M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_100M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer_1B(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data_na;
    }
  }
  *after = clock64();

}

template <typename T>
__global__ void GPUListConsumer(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    result[i] = (ptr1[order[i]]).data_na;
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_100K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_1M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_10M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_100M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_1B(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

// Loads from the array of objects in the shuffled order into the result array (which is in GDDR)
template <typename T>
__global__ void GPUListConsumer_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int * result, int * order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_100K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_1M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_10M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_100M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel_1B(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int * result, int * order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_100K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_1M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
      result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_10M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_100M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq_1B(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int * result, int * order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_1K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_10K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_100K(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_1M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < _count; i++) {
      result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
      result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_10M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 10000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_100M(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 100000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel_1B(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif
    int _count = *count;

    *before = clock64();
    for (int j = 0; j < 1000000000; j++) {
        for (int i = 0; i < _count; i++) {
            result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
            result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
        }
    }
    *after = clock64();
}

template <typename T>
__global__ void GPUListConsumer_acq_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int * result, int * order, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  int _count = *count;

  *before = clock64();
  for (int i = 0; i < _count; i++) {
    result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
  }
  *after = clock64();
}

// Stores to the array of objects in the shuffled order
template <typename T>
__global__ void GPUListProducer(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif

  for (int i = 0; i < *count; i++) {
    (ptr[order[i]]).data_na = i;
  }

    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

// Stores to the array of objects in the shuffled order
template <typename T>
__global__ void GPUListProducer_rel(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
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

template <typename T>
__host__ void CPUListConsumer(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    result[i] = (ptr[order[i]]).data_na;
    // result[i] = (ptr[order[i]]).data.load(cuda::memory_order_acquire);
  }
}

// Loads from the array of objects in the shuffled order into the result array (which is in DRAM)
template <typename T>
__host__ void CPUListConsumer_rel(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr[order[i]]).data.load(cuda::memory_order_relaxed);
  }
}

template <typename T>
__host__ void CPUListConsumer_acq(cuda::atomic<int>* flag, T* ptr, int * result, int * order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr[order[i]]).data.load(cuda::memory_order_acquire);
  }
}

template <typename T>
__host__ void CPUListProducer(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    
    #ifdef RC
    flag->store(0, cuda::memory_order_release);
    #endif
  
  for (int i = 0; i < *count; i++) {
    (ptr[order[i]]).data_na = i;
    // (ptr[order[i]]).data.store(i, cuda::memory_order_relaxed);
  }
  
    #ifdef RC
    flag->store(1, cuda::memory_order_release);
    #endif
}

// Stores to the array of objects in the shuffled order
template <typename T>
__host__ void CPUListProducer_rel(cuda::atomic<int>* flag, T* ptr, int * order, int *count) {
    
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