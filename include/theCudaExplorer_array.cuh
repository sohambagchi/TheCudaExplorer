#ifndef THECUDAEXPLORER_ARRAY_H_
#define THECUDAEXPLORER_ARRAY_H_

#include "theCudaExplorer_top.cuh"

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
__global__ void GPUListConsumer_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
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
__global__ void GPUListConsumer_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
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
__global__ void GPUListConsumer_acq_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
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
__global__ void GPUListConsumer_acq_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count, unsigned int *before, unsigned int *after) {
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
__global__ void GPUListProducer(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
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
__global__ void GPUListProducer_rel(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
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
__host__ void CPUListConsumer(cuda::atomic<int>* flag, T* ptr, int *result, int *order, int *count) {
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
__host__ void CPUListConsumer_rel(cuda::atomic<int>* flag, T* ptr, int *result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr[order[i]]).data.load(cuda::memory_order_relaxed);
  }
}

template <typename T>
__host__ void CPUListConsumer_acq(cuda::atomic<int>* flag, T* ptr, int *result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr[order[i]]).data.load(cuda::memory_order_acquire);
  }
}

template <typename T>
__host__ void CPUListConsumer_acq_rel(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_relaxed);
  }
}

template <typename T>
__host__ void CPUListConsumer_acq_acq(cuda::atomic<int>* flag, T* ptr1, T* ptr2, int *result, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  for (int i = 0; i < *count; i++) {
    // result[i] = (ptr[order[i]]).data;
    result[i] = (ptr1[order[i]]).data.load(cuda::memory_order_acquire);
    result[i] = (ptr2[order[i]]).data.load(cuda::memory_order_acquire);
  }
}

template <typename T>
__host__ void CPUListProducer(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
    
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
__host__ void CPUListProducer_rel(cuda::atomic<int>* flag, T* ptr, int *order, int *count) {
    
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


#endif /* THECUDAEXPLORER_ARRAY_H_ */