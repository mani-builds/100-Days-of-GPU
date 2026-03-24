#include <stdio.h>

#define SIZE 4096

__global__ void matMul(float *M, float *N, float *P, int width) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < SIZE && row < SIZE) {
    float sum = 0.0f;
    for (int i = 0; i < width; i++) {
    sum += M[row * width + i] * N[i * width + col];
    }
    P[row*width + col] = sum;

  }
}

int main(){
    int M_Row_h, M_Col_h, N_Row_h, N_Col_h;
    float *M_h, *N_h, *P_h;

    M_Row_h = SIZE;
    M_Col_h = SIZE;
    N_Row_h = SIZE;
    N_Col_h = SIZE;

    M_h = (float*) malloc(M_Row_h * M_Col_h * sizeof(float));
    N_h = (float*) malloc(N_Row_h * N_Col_h * sizeof(float));
    P_h = (float*) malloc(M_Row_h * N_Col_h * sizeof(float));

    for (int i = 0; i < SIZE * SIZE; i++) {
      M_h[i] = 1.0f;
      N_h[i] = 1.0f;
    }

    if (M_Col_h == N_Row_h){
        float *M_d, *N_d, *P_d;

        cudaMalloc(&M_d, M_Row_h * M_Col_h * sizeof(float));
        cudaMalloc(&N_d, N_Row_h * N_Col_h * sizeof(float));
        cudaMalloc(&P_d, M_Row_h * N_Col_h * sizeof(float));

        cudaMemcpy(M_d, M_h, M_Row_h * M_Col_h * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(N_d, N_h, N_Row_h * N_Col_h * sizeof(float), cudaMemcpyHostToDevice);


// === 2D GRID/BLOCK ===
    dim3 threadsPerBlock(32, 32);  // 2x2 = 4 threads per block
    dim3 blocksPerGrid(
        (N_Col_h + threadsPerBlock.x - 1) / threadsPerBlock.x,  // = (2+1)/2 = 2
        (M_Row_h + threadsPerBlock.y - 1) / threadsPerBlock.y   // = (3+1)/2 = 2
    );
    matMul<<<blocksPerGrid, threadsPerBlock>>>(
        M_d, N_d, P_d, SIZE
    );



        cudaMemcpy(P_h, P_d, M_Row_h * N_Col_h * sizeof(float), cudaMemcpyDeviceToHost);
         // Verify a few elements
        printf("\nFirst 10: \n");
      for (int i = 0; i < 10; ++i){
          printf("%f\n", P_h[i]);
      }
   printf("\nLast 10: \n");
      for (int i = SIZE*SIZE-10; i < SIZE*SIZE; ++i){
          printf("%f\n", P_h[i]);
      }


      cudaFree(M_d); cudaFree(N_d); cudaFree(P_d);
      free(M_h); free(N_h); free(P_h);


    }
    else{
        printf("Dimensions doesn't match, canoot multiply\n");
    }

    return 0;
}
