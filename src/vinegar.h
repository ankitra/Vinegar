#include "cuda.h"
#define CHECKED_CUDA_API(api) { \
    cudaError_t err = api; \
    if (error != cudaSuccess) { \
    printf(“%s in %s at line %d\n”, cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
   } \
}

