#include <stdio.h>

// Tiled 7 point stencil (with order = 1)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

#define IN_TILE_DIM 4
#define OUT_TILE_DIM ((IN_TILE_DIM) - (2))

__global__ void tiled_stencil_kernel(float *in, float *out, unsigned int N) {
  // Start indexing from the starting point of the stencil patch
  int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  // load the input tile
  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  // The boundary rows and cols of input and output are the same i.e stencil doesn't apply on those.
  if (i >= 0 && i < N  && j >= 0 && j < N  && k >= 0 && k < N ) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
  }
  __syncthreads();
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
        threadIdx.x < IN_TILE_DIM - 1) {
      out[i * N * N + j * N + k] =
          c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
          c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] +
          c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] +
          c3 * in_s[threadIdx.z][threadIdx.y-1][threadIdx.x] +
          c4 * in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] +
          c5 * in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] +
          c6 * in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
    }
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
  printf("INPUT: (1st N) \n");
  for (int i = 0; i < N; i++) {
    printf("%f \t", input_h[i]);
    }
    printf("\n");


  float *input, *output;

  cudaMalloc(&input, N*N*N*sizeof(float));
  cudaMalloc(&output, N*N*N*sizeof(float));

  cudaMemcpy(input, input_h, N*N*N*sizeof(float), cudaMemcpyHostToDevice);

  printf("\nInput tile : %d and output tile: %d dim\n ", IN_TILE_DIM, OUT_TILE_DIM);

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

  dim3 threadsPerBlock(8,8,8);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (N + threadsPerBlock.z - 1) / threadsPerBlock.z);


  tiled_stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaMemcpy(output_h, output, N*N*N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("OUTPUT: (1st N)\n");
  for (int i = 0; i < N; i++) {
    printf("%f \t", output_h[i]);
    }
    printf("\n");


  cudaFree(input);
  cudaFree(output);
  free(input_h);
  free(output_h);


  return 0;
}
