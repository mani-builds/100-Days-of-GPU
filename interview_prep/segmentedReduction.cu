#include <cmath>
#include <stdio.h>

#define SIZE 4096
#define BLOCK_SIZE 1024

__global__ void simple_reduction_kernel(float *input, int N, float *output) {

  // We can more than single thread blocks, so each block
  // evaluates a segment of size 2*blockDIM (since each thread handles 2 data).
  // Therefore there are multiple blocks, but we use 'Atomic' operation to sync output
  // at the end

  int segment = 2 * blockDim.x * blockIdx.x;
  int i = segment + threadIdx.x;

  __shared__ float input_s[BLOCK_SIZE];

  if (threadIdx.x < BLOCK_SIZE && i + BLOCK_SIZE < N){
  input_s[threadIdx.x] = input[i]+ input[i + BLOCK_SIZE]; // parital reduce to BLOCK_SIZE
  } else {
    input_s[threadIdx.x] = 0.0f;

  }
  // Since BLOCK_SIZE = 1024 (max), stride = blockDim.x/2
  // apply parallel reduction on the smem of size BLOCK_SIZE
  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
    __syncthreads(); // This takes care of synching for the shared mem too, during
                     // first iteration
    if (threadIdx.x < stride){
    input_s[threadIdx.x] = input_s[threadIdx.x]+ input_s[threadIdx.x+stride];
    }
  }

  if (threadIdx.x == 0) {
    atomicAdd(output ,input_s[threadIdx.x]);
  }

}

int main() {
  float *in_h;
  float *out_h;

  int N = SIZE;

  in_h = (float *) malloc(N * sizeof(float));
  out_h = (float *) malloc(sizeof(float));

  for(int i=0; i<N; i++) in_h[i] = 1;

  float *in;
  float *out;

  cudaMalloc(&in, N*sizeof(float));
  cudaMalloc(&out, sizeof(float));

  cudaMemcpy(in, in_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(out, 0, sizeof(float));
  int threads = 1024;
  int blocksPerGrid = (N/2 + threads - 1) / threads;
  simple_reduction_kernel<<<blocksPerGrid, threads>>>(in, N, out);

  cudaMemcpy(out_h, out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Output: %f \n", *out_h);

  free(in_h);
  cudaFree(in);
  cudaFree(out);

  return 0;
}
