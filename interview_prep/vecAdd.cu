#include <stdio.h>

#define N 4096

__global__ void vecAdd(float *A, float *B, float *C, int n) {

  // int i = blockDim.x * blockIdx.x + threadIdx.x;
  // if (i < n)
  // C[i] = A[i] + B[i];

  // if single thread block
  int section_size = N / blockDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // __shared__ float a_s[1024];
  // __shared__ float b_s[1024];

  for (int k = 0; k < section_size; k++) {
    C[i + k*blockDim.x] = A[i + k*blockDim.x] + B[i + k*blockDim.x];
  }

}

int main() {
    int n = N;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1; //(n + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    printf("Result (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    printf("Result (last 10 elements):\n");
    for (int i = N-10; i < N; ++i) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
