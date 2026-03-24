

#define TILE_WIDTH 32
__global__ void kernel(float *M, float *N, float *P, int width) {

  // Map threads
  int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
  int col = TILE_WIDTH * blockIdx.x + threadIdx.x;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
      // smem
      float Pvalue = 0.0f;
  __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

      // Perform matmul
  for (int ph = 0; ph < width / TILE_WIDTH; ph++) {

    if ( row  < width && (ph * TILE_WIDTH + tx)){
    Ms[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
    } else {
      Ms[ty][tx]  = 0.f;

    }
    if (col < width &&  (ph * TILE_WIDTH + ty) < width){
    Ns[ty][tx] = N[(ph * TILE_WIDTH + ty)*width + col];
    } else
      Ns[ty][tx]  = 0.f;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      Pvalue += Ms[ty][k] * Ns[k][tx];
    }

    __syncthreads();
  }

  // HBM
  if (row < width && col < width)
  P[row * width + col] = Pvalue;
}
