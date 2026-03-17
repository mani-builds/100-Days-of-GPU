#include <stdio.h>

#define BLOCK_DIM 1024

__global__ void SimpleSumReductionKernel(float *input, float *output) {

  __shared__ float input_s[BLOCK_DIM];
  unsigned int tid = threadIdx.x;
  input_s[tid] = input[tid] + input[tid+BLOCK_DIM]; // Load and add the 1st iteration data from Global to shared

  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) { // 1st iteration is already done, so blockDim.x/2
    __syncthreads(); // move synchronize to the start of 'for' loop, so that the initial
                     // load from Global to Shared Mem happens completely
    if (threadIdx.x < stride) {
      input_s[tid] += input_s[tid + stride];
    }
  }

  if (threadIdx.x == 0) {
    *output = input_s[0];
  }
}

int main() {

  float *input_h;
  float *output_h;
  float init_value = 0.0f;
  output_h = &init_value;

  int N = 2*1024;

  printf("Intial value: %f \n", *output_h);
  input_h = (float *)malloc(N * sizeof(float));
  // memset(input_h, 1.0, N*sizeof(float));
  for(int i=0; i <N; i++){ input_h[i]=1.0f;}
  float *input;
  float *output;

  cudaMalloc(&input, N * sizeof(float));
  cudaMemcpy(input, input_h, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&output, sizeof(float));
  int threads = BLOCK_DIM;
  int blocks = (1 + threads - 1) / threads;
  SimpleSumReductionKernel<<<blocks, threads>>>(input,output);

  cudaMemcpy(output_h, output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("FInal value: %f\n", *output_h);
  cudaFree(input);
  cudaFree(output);
  free(input_h);
  return 0;
}
