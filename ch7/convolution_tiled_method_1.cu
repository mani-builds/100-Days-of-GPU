# include <stdio.h>

#define FILTER_RADIUS 1
__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

#define IN_TILE_DIM 5
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height){
  int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
  // loading input tiles
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  if (row >=0 && row < height && col >=0 && col <width){
  N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
}
  else{
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
__syncthreads();
// calculating output threads
int tileCol = threadIdx.x - FILTER_RADIUS;
int tileRow = threadIdx.y - FILTER_RADIUS;
// turning off the threads at the edge
  if (row >=0 && row < height && col >=0 && col <width){
  if (tileRow >=0 && tileRow  < OUT_TILE_DIM && tileCol >=0 && tileCol < OUT_TILE_DIM){
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow<2*FILTER_RADIUS+1; fRow++){
      for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++){
        Pvalue += F[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
      }
    }
    P[row*width + col] = Pvalue;
  }
  }


}

int main(){
  float *N_h, *P_h;
  int width, height;
  int r = FILTER_RADIUS;
  float F_h[2*r+1][2*r+1];

  width = 8;
  height = 8;

  N_h = (float *)malloc(sizeof(float) * width * height);
  P_h = (float *)malloc(sizeof(float) * width * height);

  int count = 0;

  for (int i = 0; i < width*height; i++){
    N_h[i] = count++;
  }

  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
    if ((i+j) % 2 == 0){ F_h[i][j] = 0;}
    else { F_h[i][j] = 1;}
  }
  }

  printf("\nInput: \n");
  for (int i = 0; i < width*height; i++){
    printf("%f \t", N_h[i]);
  }
  printf("\nFilter: \n");
  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
    printf("%f \t", F_h[i][j]);
  }
  }

  float *N, *P;
  cudaMalloc(&N, sizeof(float) * width * height);
  cudaMalloc(&P, sizeof(float) * width * height);

  cudaMemcpy(N, N_h,sizeof(float) * width * height, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F,F_h,sizeof(float) * (2*r+1) * (2*r+1));

  dim3 threadsPerBlock(5,5);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_tiled_2D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,P,width,height);
  cudaMemcpy(P_h, P,sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height; i++) {
        printf("%f\n", P_h[i]);
    }

    cudaFree(P);
    cudaFree(N);
    free(N_h);
    free(P_h);


  return 0;
}
