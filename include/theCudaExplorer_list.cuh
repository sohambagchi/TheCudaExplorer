#ifndef THECUDAEXPLORER_LIST_H_
#define THECUDAEXPLORER_LIST_H_

#include "theCudaExplorer_top.cuh"

template <typename T>
__global__ void GPULinkedListInit(cuda::atomic<int>* flag, T *head, int *order, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  int _count = *count;

  // for (int i = 0; i < _count; i++) {
  //   head[order[i]].data_na = i;
  //   head[order[i]].data.store(i);
  // }

  for (int i = 0; i < _count; i++) {
    head[order[i]].data.store(order[(i + 1) % _count]);
    head[order[i]].data_na = order[(i + 1) % _count];
  }

}

template <typename T>
__global__ void GPULinkedListConsumer_acq(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int i = 0; i < *count; i++) {
    next1 = current1[next1].data.load(cuda::memory_order_acquire);
    result[i] = next1;
    // next2 = current2[next2].data.load(cuda::memory_order_relaxed);
    // result[i] = next2;
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_1K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2;
  int next1 = 0;
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000; j++) {
    // next1 = 0;
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_10K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_100K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_1M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_10M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_100M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  // T* current2 = head2; 
  int next1 = 0; 
  // int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      // next2 = current2[next2].data.load(cuda::memory_order_acquire);
      // result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int i = 0; i < *count; i++) {
    // next1 = current1[next1].data.load(cuda::memory_order_acquire);
    // result[i] = next1;
    next2 = current2[next2].data.load(cuda::memory_order_relaxed);
    result[i] = next2;
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_1K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_10K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_100K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_1M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_10M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_rel_100M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  // T* current1 = head1;
  T* current2 = head2; 
  // int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < *count; i++) {
      // next1 = current1[next1].data.load(cuda::memory_order_acquire);
      // result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int i = 0; i < *count; i++) {
    next1 = current1[next1].data.load(cuda::memory_order_acquire);
    result[i] = next1;
    next2 = current2[next2].data.load(cuda::memory_order_acquire);
    result[i] = next2;
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_1K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_10K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_100K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_1M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_10M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_acq_100M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_acquire);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int i = 0; i < *count; i++) {
    next1 = current1[next1].data.load(cuda::memory_order_acquire);
    result[i] = next1;
    next2 = current2[next2].data.load(cuda::memory_order_relaxed);
    result[i] = next2;
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_1K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
    #ifdef RC
    while (flag->load(cuda::memory_order_acquire) == 0) {}
    #endif

    T* current1 = head1;
    T* current2 = head2; 
    int next1 = 0; 
    int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_10K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_100K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_1M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_10M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_acq_rel_100M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif

  T* current1 = head1;
  T* current2 = head2; 
  int next1 = 0; 
  int next2 = 0;

  *before = clock64();

  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < *count; i++) {
      next1 = current1[next1].data.load(cuda::memory_order_acquire);
      result[i] = next1;
      next2 = current2[next2].data.load(cuda::memory_order_relaxed);
      result[i] = next2;
    }
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  
  for (int i = 0; i < *count; i++) {
    next = current[next].data_na;
    result[i] = next;
    // next = current[next].next;
  }

  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_1K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_10K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 10000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_100K(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 100000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_1M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 1000000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_10M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 10000000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__global__ void GPULinkedListConsumer_100M(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count, unsigned int *before, unsigned int *after) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head1; int next = 0;

  *before = clock64();
  for (int j = 0; j < 100000000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
  *after = clock64();
}

template <typename T>
__host__ void CPULinkedListConsumer(cuda::atomic<int>* flag, T* head, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head; int next = 0;
  for (int i = 0; i < *count; i++) {
    next = current[next].data_na;
    result[i] = next;
    // next = current[next].next;
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_1K(cuda::atomic<int>* flag, T* head, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head; int next = 0;
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data_na;
      result[i] = next;
      // next = current[next].next;
    }
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_acq(cuda::atomic<int>* flag, T* head, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head; int next = 0;
  for (int i = 0; i < *count; i++) {
    next = current[next].data.load(cuda::memory_order_acquire);
    result[i] = next;
    // next = current[next].next;
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_acq_1K(cuda::atomic<int>* flag, T* head, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head; int next = 0;
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < *count; i++) {
      next = current[next].data.load(cuda::memory_order_acquire);
      result[i] = next;
      // next = current[next].next;
    }
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_rel(cuda::atomic<int>* flag, T* head, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current = head; int next = 0;
  for (int i = 0; i < *count; i++) {
    next = current[next].data.load(cuda::memory_order_relaxed);
    result[i] = next;
    // next = current[next].next;
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_acq_rel(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current1 = head1; int next1 = 0;
  T* current2 = head2; int next2 = 0;
  for (int i = 0; i < *count; i++) {
    next1 = current1[next1].data.load(cuda::memory_order_acquire);
    result[i] = next1;
    next2 = current2[next2].data.load(cuda::memory_order_relaxed);
    result[i] = next2;
    // next = current[next].next;
  }
}

template <typename T>
__host__ void CPULinkedListConsumer_acq_acq(cuda::atomic<int>* flag, T* head1, T* head2, int *result, int *count) {
  #ifdef RC
  while (flag->load(cuda::memory_order_acquire) == 0) {}
  #endif
  T* current1 = head1; int next1 = 0;
  T* current2 = head2; int next2 = 0;
  for (int i = 0; i < *count; i++) {
    next1 = current1[next1].data.load(cuda::memory_order_acquire);
    result[i] = next1;
    next2 = current2[next2].data.load(cuda::memory_order_acquire);
    result[i] = next2;
    // next = current[next].next;
  }
}

template <typename T>
__global__ void GPULinkedListProducer(cuda::atomic<int>* flag, T* head, int *order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current[order[i]].data_na = order[(i + 1) % *count];
    // next = current[next].next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}
 
template <typename T>
__global__ void GPULinkedListProducer_rel(cuda::atomic<int>* flag, T* head, int *order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current[order[i]].data.store(order[(i + 1) % *count], cuda::memory_order_relaxed);
    // next = current[next].next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPULinkedListProducer(cuda::atomic<int>* flag, T* head, int *order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current[order[i]].data_na = order[(i + 1) % *count];
    // next = current[next].data_na;
    // current[next].data_na = next;
    // next = current[next].next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

template <typename T>
__host__ void CPULinkedListProducer_rel(cuda::atomic<int>* flag, T* head, int *order, int *count) {
  T* current = head;
  for (int i = 0; i < *count; i++) {
    current[order[i]].data.store(order[(i + 1) % *count], cuda::memory_order_relaxed);
    // next = current[next].next;
  }
  #ifdef RC
  flag->store(1, cuda::memory_order_release);
  #endif
}

#endif /* THECUDAEXPLORER_LIST_H_ */