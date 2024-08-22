#ifndef THECUDAEXPLORER_MIGRATE_H_
#define THECUDAEXPLORER_MIGRATE_H_

#include "theCudaExplorer_top.cuh"


template <typename T>
__global__ void gpuMigrationExperiment(T * head, int * outerCount, int * firstCount, int * secondCount, int * results) {
    T * current = head;
    
    for (int j = 0; j < *firstCount; j++) { 
        for (int i = 0; i < *outerCount; i++) {
            // top_data is at the start of the page -- first 256 accesses
            results[i * *outerCount + j] = current[i].top_data[j].load(cuda::memory_order_relaxed);
            // effective latency = 680ns
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            // bottom_data is at end of page -- this loop does 128 accesses, and the next loop does 128 more
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
            // effective latency = 180ns
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            // this loop does 128 accesses
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__host__ void cpuMigrationExperiment(T * head, int * outerCount, int * firstCount, int * secondCount, int * results) {
    T * current = head;
    
    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *firstCount; j++) {
            results[i * *outerCount + j] = current[i].top_data[j].load(cuda::memory_order_relaxed);
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__global__ void gpuMigrationExperiment_na(T * head, int * outerCount, int * firstCount, int * secondCount, int * results) {   
    T * current = head;
    T * current1 = head;
    
    for (int j = 0; j < *firstCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].top_data[j];
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].bottom_data[j];
        }
    }

    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current1[i].bottom_data[j];
        }
    }
}

template <typename T>
__host__ void cpuMigrationExperiment_na(T * head, int * outerCount, int * firstCount, int * secondCount, int * results) {
    T * current = head;
    T * current1 = head;
    
    for (int j = 0; j < *firstCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].top_data[j];
        }
    }
    
    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current[i].bottom_data[j];
        }
    }

    for (int j = 0; j < *secondCount; j++) {
        for (int i = 0; i < *outerCount; i++) {
            results[i * *outerCount + j] = current1[i].bottom_data[j];
        }
    }
}

template <typename T>
__global__ void gpuFillAccessCounter_store(T * head, int * outerCount, int * innerCount) {
        
    T * current = head;
    
    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            current[i].top_data[j].fetch_add(i * j, cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__global__ void gpuFillAccessCounter(T * head, int * outerCount, int * innerCount, int * results) {

    T * current = head;
    
    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].top_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__global__ void gpuCheckMigration_rel(T * head, int * outerCount, int * innerCount, int * results) {
    
    T * current = head;

    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__global__ void gpuCheckMigration_acq(T * head, int * outerCount, int * innerCount, int * results) {
        
    T * current = head;

    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_acquire);
        }
    }
}

template <typename T>
__host__ void cpuFillAccessCounter_store(T * head, int * outerCount, int * innerCount) {
    
    T * current = head;
    
    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            current[i].top_data[j].fetch_add(i * j, cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__host__ void cpuFillAccessCounter(T * head, int * outerCount, int * innerCount, int * results) {
    
    T * current = head;
    
    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].top_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__host__ void cpuCheckMigration_rel(T * head, int * outerCount, int * innerCount, int * results) {
    
    T * current = head;

    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_relaxed);
        }
    }
}

template <typename T>
__host__ void cpuCheckMigration_acq(T * head, int * outerCount, int * innerCount, int * results) {
    
    T * current = head;

    for (int i = 0; i < *outerCount; i++) {
        for (int j = 0; j < *innerCount; j++) {
            results[i * *outerCount + j] = current[i].bottom_data[j].load(cuda::memory_order_acquire);
        }
    }
}

#endif // THECUDAEXPLORER_MIGRATE_H_
