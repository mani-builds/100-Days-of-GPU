#include <stdio.h>

#define SIZE 2048

__global__ void simple_reduction_kernel(float *input, int N, float *output) {
  int i = threadIdx.x;

  // Optimization 1:
  // Store the data in shared mem. Each thread loads two elements into the s mem.
  // Since it's single thread block, shared mem should have space for all 2048 data
  __shared__ float input_s[SIZE];
  input_s[i] = input[i];
  // input_s[i + blockDim.x] = input[i+blockDim.x];
  __syncthreads();

  // Optimization 2:
  // By having stride = blockDim.x/2, we reduce control divergence within
  // the warps themselves until the last iteration of 'for' loop.
  // This also helps us with memory coalescing, since each thread load elements
  // relative to the thread indices
  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride){
    input_s[i] = max(input_s[i], input_s[i+stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = input_s[threadIdx.x];
  }

}

int main() {
  float *in_h;
  float *out_h;

  int N = SIZE;

  in_h = (float *) malloc(N * sizeof(float));
  out_h = (float *) malloc(sizeof(float));

  for(int i=0; i<N; i++) in_h[i] = i;

  float *in;
  float *out;

  cudaMalloc(&in, N*sizeof(float));
  cudaMalloc(&out, sizeof(float));

  cudaMemcpy(in, in_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = N/2;
  int blocksPerGrid = 1;
  simple_reduction_kernel<<<blocksPerGrid, threads>>>(in, N, out);

  cudaMemcpy(out_h, out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Output: %f \n", *out_h);

  free(in_h);
  cudaFree(in);
  cudaFree(out);

  return 0;
}
