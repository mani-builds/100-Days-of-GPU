#include <cmath>
#include <stdio.h>

#define SIZE 4096

__global__ void simple_reduction_kernel(float *input, int N, float *output) {
  int i = threadIdx.x;

  __shared__ float input_s[1024];

  // Optimization
  // We can further improve execution speed by keeping the partial results (reductions)
  // in the shared memory
  float max_value = -INFINITY;
  for(int offset=0; offset <SIZE/blockDim.x ; offset++){
   max_value = max(max_value, input[i + offset*blockDim.x]);
  }
  input_s[i] = max_value;

  // This further decreases the need to have large shared mem, and also processes
  // more than the 'double' the blockDim number of elements, by keeping the partial
  // results in shared mem of size blockDim (Note the SIZE = 4096 and blockDim = 1024)
  // We need to have stride = blockDim.x/2
  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
    __syncthreads(); // This takes care of synching for the shared mem too during
                     // first iteration
    if (threadIdx.x < stride){
    input_s[i] = max(input_s[i], input_s[i+stride]);
    }
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

  int threads = 1024;
  int blocksPerGrid = 1;
  simple_reduction_kernel<<<blocksPerGrid, threads>>>(in, N, out);

  cudaMemcpy(out_h, out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Output: %f \n", *out_h);

  free(in_h);
  cudaFree(in);
  cudaFree(out);

  return 0;
}
