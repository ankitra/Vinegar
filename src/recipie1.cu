#include "vinegar.h"

__global__
void vecAddKernel(float * A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n)
    C[i] = A[i] + B[i];
}

void vecAdd(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  size_t size = sizeof(float[n]);

  CHECKED_CUDA_API(cudaMalloc((void **) &A_d,size));
  CHECKED_CUDA_API(cudaMalloc((void **) &B_d,size));
  CHECKED_CUDA_API(cudaMalloc((void **) &C_d,size));

  CHECKED_CUDA_API(cudaMemcpy(A_d,A,size,cudaMemcpyHostToDevice));
  CHECKED_CUDA_API(cudaMemcpy(B_d,B,size,cudaMemcpyHostToDevice));

  vecAddKernel <<<ceil(n/256.0), 256>>>(A_d,B_d,C_d,n);

  CHECKED_CUDA_API(cudaMemcpy(C,C_d,size,cudaMemcpyDeviceToHost));

  CHECKED_CUDA_API(cudaFree(A_d));
  CHECKED_CUDA_API(cudaFree(B_d));
  CHECKED_CUDA_API(cudaFree(C_d));

}

