/*
 * Vinegar for recipies, helps to cook faster
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"

#ifndef _RED_DOT_VINEGAR_DOT_H_
#define _RED_DOT_VINEGAR_DOT_H_

// Call Cuda and exit if it did not work.
#define CHECKED_CUDA_API(api)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = api;                                                                     \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            fflush(stderr);                                                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// Dimensions of threads and blocks and grids
#define X_DIM x
#define Y_DIM y
#define Z_DIM z

// Offset in current thread
#define THREAD_OFFSET(dimension) (threadIdx.dimension + blockDim.dimension * blockIdx.dimension)

// ANSI C makes sizeof(char) as always 1 byte, irrespective of what processor and architecture is.
// It will change definition of 1 byte to accomodate that.
#define ROW_MAJOR(row, col, base, row_elements, type) (type *)((char *)(base) + (((row) * (row_elements) + (col)) * (sizeof(type))))
#define COL_MAJOR(row, col, base, col_elements, type) (type *)((char *)(base) + (((col) * (col_elements) + (row)) * (sizeof(type))))

// Debug logging
#ifdef DEBUG
    #define DEBUG_LOG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...)
#endif

#define ERROR_LOG(fmt, ...) do { \
    fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); \
    fflush(stderr); \
} while (0)

// Instrument the kernel execution time, report the metrics on stderr
#ifdef INSTRUMENTED

    #define TIME_CUDA(message, cuda_invoke)                            \
        do                                                             \
        {                                                              \
            cudaEvent_t start, stop;                                   \
            float elapsedTime;                                         \
            CHECKED_CUDA_API(cudaEventCreate(&start));                 \
            CHECKED_CUDA_API(cudaEventCreate(&stop));                  \
            CHECKED_CUDA_API(cudaEventRecord(start, 0));               \
            cuda_invoke;                                               \
            CHECKED_CUDA_API(cudaEventRecord(stop, 0));                \
            cudaEventSynchronize(stop);                                \
            cudaEventElapsedTime(&elapsedTime, start, stop);           \
            DEBUG_LOG("Time elapsed %f for %s", elapsedTime, message); \
            fflush(stderr);                                            \
            CHECKED_CUDA_API(cudaEventDestroy(start));                 \
            CHECKED_CUDA_API(cudaEventDestroy(stop));                  \
        } while (0)

#else

    #define TIME_CUDA(message, cuda_invoke) cuda_invoke

#endif

#endif