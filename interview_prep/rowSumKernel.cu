#include <stdio.h>

#define M 1024
#define N 4096

__global__ void rowSumKernel(const float *mat,
                             float* vec, int m, int n) {

  int i = threadIdx.x; // data index
  // Shared mem
  __shared__ float x_s[1024];

  for (int row = 0; row < m; row ++){

    // Have an accumulator which is unique to threads since it's in register
    // to store the intermediate sum
    float sum = 0.0f;
    for (int k = 0; k < N / blockDim.x; k++){ // Iterates for 4 time if N = 4096
    sum += mat[row * n + i + k*blockDim.x];
    }
    x_s[i] = sum; // Helps with not needing explicit flush
  __syncthreads();

  // Reduction
  for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride){
   x_s[i] = x_s[i] + x_s[i + stride];
    }
    __syncthreads();
  }

  // HBM
  if (threadIdx.x == 0)
  vec[row] = x_s[threadIdx.x];
}
  // Ensure all threads are ready before moving to the next row
  __syncthreads();
}


int main() {
  float *mat_h;
  float *vec_h;
  int m, n;

  m = M;
  n = N;

  mat_h = (float *) malloc(m*n*sizeof(float));
  vec_h = (float *) malloc(m*sizeof(float));

  for(int i=0; i< m*n ; i++) mat_h[i] = 1.0f;
  for(int i=0; i< m ; i++) vec_h[i] = 0.0f;

  printf("\nMat array : \n");
  for(int i=0; i< 10 ; i++) printf("%f \t", mat_h[i]);
  printf("\n");

  float *mat;
  float *vec;

  cudaMalloc(&mat, m * n * sizeof(float));
  cudaMalloc(&vec, m*sizeof(float));

  cudaMemcpy(mat, mat_h, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vec, vec_h, m*sizeof(float), cudaMemcpyHostToDevice);

  // Contraint: single thread block and limited Shared Mem (so cannot fit the entire matrix)
  int threads = 1024; // each thread handles addition of 2 elems (Launch threads to work on each col)
  int blocks = 1;

  printf("\n blocks: %d, threads: %d\n", blocks, threads);

  rowSumKernel<<<blocks, threads>>>(mat, vec, m,n);

  cudaMemcpy(vec_h, vec, m*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nVec array : \n");
  for(int i=0; i< 10 ; i++) printf("%f \t", vec_h[i]);
  printf("\n");

  free(mat_h);
  free(vec_h);
  cudaFree(mat);
  cudaFree(vec);
}
