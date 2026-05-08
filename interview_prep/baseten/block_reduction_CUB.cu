#include <stdio.h>
#include <cub/cub.cuh> // Using CUDA Unbound library

#define BLOCK_SIZE 512

__global__ void warp_reduce_kernel(float *in, float *out, int N) {

  // Map data to threads
  int i = blockIdx.x * blockIdx.x * 2 + threadIdx.x; // Skip each other block

  // Specialize BlockReduce for a 1D block of 512 threads on type float
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

  // Allocate shared memory for CUB collective
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float sum = 0;

  if (i < N) sum += in[i];
  if (i + BLOCK_SIZE < N){
  sum += in[i + BLOCK_SIZE]; // Add two elements (the other from the next block)
  }

  // logic Blockwise reduction
  float aggregate = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      // *out = val;
      // For large arrays
      atomicAdd(out, aggregate);
    }
}


int main() {
  float *in_h;
  float *out_h;

  int N = 4096;

  in_h = (float *) malloc(N * sizeof(float));
  out_h = (float *) malloc(sizeof(float));

  for(int i=0; i<N; i++) in_h[i] = 1;

  float *in;
  float *out;

  cudaMalloc(&in, N*sizeof(float));
  cudaMalloc(&out, sizeof(float));

  cudaMemcpy(in, in_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = BLOCK_SIZE;
    // Each block processes 2 * BLOCK_SIZE elements
  int blocksPerGrid = (N + (2*threads) - 1)/ (2*threads);
  warp_reduce_kernel<<<blocksPerGrid, threads>>>(in, out, N);

  printf("blocks: %d, threads: %d\n", blocksPerGrid, threads);
  cudaMemcpy(out_h, out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Output: %f \n", *out_h);

  free(in_h);
  cudaFree(in);
  cudaFree(out);

  return 0;
}
