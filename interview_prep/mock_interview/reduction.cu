#include <stdio.h>

#define N 4096
#define BLOCK 1024

__global__ void reduction(float *a, float *o, int n) {
  // Each thread adds two elements
  int i = 2*blockIdx.x * blockDim.x + threadIdx.x;

  // smem
  __shared__ float as[BLOCK];
  float sum = 0;
  for(int k=0; k< n/BLOCK ; k++){
    sum += a[i + k*BLOCK];
  }
  as[threadIdx.x] = sum;

  //logic
   for (int stride = BLOCK / 2; stride >= 1; stride /= 2) {
     __syncthreads();
    if(threadIdx.x < stride){
     as[threadIdx.x] = as[threadIdx.x] + as[threadIdx.x + stride];
     }
   }

  //HBM
  if (threadIdx.x == 0){
    atomicAdd(o, as[0]);
  }
}


int main() {
    // 1. Allocate and initialize host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // or any pattern you like
    }

    float h_out = 0.0f;

    // 2. Allocate device memory
    float *d_A, *d_out;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // 3. Copy input to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch kernel
    int grid = N/2/BLOCK;
    printf("Grid: %d\n", grid);
    reduction<<<1, BLOCK>>>(d_A, d_out, N);

    // 5. Copy result back to host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Print result
    printf("Reduction result: %f\n", h_out);

    // 7. Free memory
    cudaFree(d_A);
    cudaFree(d_out);
    free(h_A);

    return 0;
}
