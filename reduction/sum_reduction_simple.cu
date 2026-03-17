#include <stdio.h>

// Reduced control divergence and improved memory coalescing

__global__ void SimpleSumReductionKernel(float *input, float *output) {
  unsigned int i = 2*threadIdx.x; // index for the data location
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads(); // to synchronize partial sums across interations
  }

  if (threadIdx.x == 0) {
    *output = input[0];
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
  int threads = N / 2;
  int blocks = (1 + threads - 1) / threads;
  SimpleSumReductionKernel<<<blocks, threads>>>(input,output);

  cudaMemcpy(output_h, output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("FInal value: %f\n", *output_h);
  cudaFree(input);
  cudaFree(output);
  free(input_h);
  return 0;
}
