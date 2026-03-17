#include <stdio.h>

__constant__ float c0,c1,c2,c3,c4,c5,c6;


__global__ void stencil_basic_kernel(float *input, float *output,
                                     unsigned int N) {
  int i =  blockIdx.z * blockDim.z + threadIdx.z;
  int j =  blockIdx.y * blockDim.y + threadIdx.y;
  int k =  blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    output[i * N * N + j * N + k] = c0 * input[i * N * N + j * N + k] +
                                    c1 * input[i * N * N + j * N + (k - 1)] +
                                    c2 * input[i * N * N + j * N + (k + 1)] +
                                    c3 * input[i * N * N + (j - 1)* N + k] +
                                    c4 * input[i * N * N + (j + 1)* N + k] +
                                    c5 * input[(i - 1)* N * N + j * N + k] +
                                    c6 * input[(i + 1)* N * N + j * N + k];
  }
}

int main() {
  float *input_h, *output_h;
  int N = 8;

  input_h = (float *)malloc(N*N*N*sizeof(float));
  output_h = (float *)malloc(N*N*N*sizeof(float));

  for (int i = 0; i < N * N * N; i++) {
    if (i % 2 == 0) {
      input_h[i] = 1.0f;
    } else {
      input_h[i] = 0.0f;
    }
  }
  printf("INPUT (just 1st N): \n");
  for (int i = 0; i < N ; i++) {
    printf("%f \t", input_h[i]);
    }
    printf("\n");


  float *input, *output;

  cudaMalloc(&input, N*N*N*sizeof(float));
  cudaMalloc(&output, N*N*N*sizeof(float));

  cudaMemcpy(input, input_h, N*N*N*sizeof(float), cudaMemcpyHostToDevice);

  float c0_h = 1.0f;
  float c1_h = 1.0f;
  float c2_h = 1.0f;
  float c3_h = 1.0f;
  float c4_h = 1.0f;
  float c5_h = 1.0f;
  float c6_h = 1.0f;

  cudaMemcpyToSymbol(&c0,&c0_h,sizeof(float));
  cudaMemcpyToSymbol(&c1,&c1_h,sizeof(float));
  cudaMemcpyToSymbol(&c2,&c2_h,sizeof(float));
  cudaMemcpyToSymbol(&c3,&c3_h,sizeof(float));
  cudaMemcpyToSymbol(&c4,&c4_h,sizeof(float));
  cudaMemcpyToSymbol(&c5,&c5_h,sizeof(float));
  cudaMemcpyToSymbol(&c6,&c6_h,sizeof(float));

  dim3 threadsPerBlock(N,N,N);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (N + threadsPerBlock.z - 1) / threadsPerBlock.z);


  stencil_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaMemcpy(output_h, output, N*N*N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("OUTPUT (first N): \n");
  for (int i = 0; i < N ; i++) {
    printf("%f \t", output_h[i]);
    }
    printf("\n");


  cudaFree(input);
  cudaFree(output);
  free(input_h);
  free(output_h);

return 0;

}
