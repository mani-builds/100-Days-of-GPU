#include <stdio.h>

#include <stdlib.h>
#include <math.h>

__global__ void softmax(float *a, int N, float *sum, float *o) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  float local = 0.0f;
  local = expf(a[row]);

  atomicAdd(sum, local);

  // HBM
  o[row] = local / *sum;
}



int main() {
    // int N = 8;
    // float h_a[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    // float h_sum = 0.0f;
    // float h_o[8];
    int N = 2000;
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_o = (float*)malloc(N * sizeof(float));
    float h_sum = 0.0f;

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 10);
    }

    float *d_a, *d_sum, *d_o;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_o, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch kernel
    softmax<<<blocks, threads>>>(d_a, N, d_sum, d_o);

    // Copy result back
    cudaMemcpy(h_o, d_o, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Softmax output (first 16):\n");
    for (int i = 0; i < 16; i++) {
        printf("%f ", h_o[i]);
    }
    printf("\n");

    free(h_a);
    free(h_o);
    cudaFree(d_a);
    cudaFree(d_sum);
    cudaFree(d_o);

    return 0;
}
