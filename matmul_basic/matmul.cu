#include <stdio.h>
#include <math.h>
__global__
void MatrixMulKernel(float *M, float *N, float *P,  int M_Row_Dim, int N_Col_Dim, int CommonDim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row <  M_Row_Dim) && (col < N_Col_Dim)){
        float element_value = 0.0;
        for (int i = 0; i<CommonDim; i++){

            element_value += M[CommonDim*row + i] * N[N_Col_Dim*i + col];

        }
        P[row*N_Col_Dim + col] = element_value;
    }
}

int main(){
    int M_Row_h, M_Col_h, N_Row_h, N_Col_h;
    float *M_h, *N_h, *P_h;

    M_Row_h = 3;
    M_Col_h = 3;
    N_Row_h = 3;
    N_Col_h = 2;

    M_h = (float*) malloc(M_Row_h * M_Col_h * sizeof(float));
    N_h = (float*) malloc(N_Row_h * N_Col_h * sizeof(float));
    P_h = (float*) malloc(M_Row_h * N_Col_h * sizeof(float));

    float M_h_temp[] = {1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0};
    float N_h_temp[] = {1.0, 3.0, 4.0, 0.0, 0.0, 1.0};
    memcpy(M_h, M_h_temp, M_Row_h * M_Col_h * sizeof(float));
    memcpy(N_h, N_h_temp, N_Row_h * N_Col_h * sizeof(float));



    if (M_Col_h == N_Row_h){
        float *M_d, *N_d, *P_d;

        cudaMalloc(&M_d, M_Row_h * M_Col_h * sizeof(float));
        cudaMalloc(&N_d, N_Row_h * N_Col_h * sizeof(float));
        cudaMalloc(&P_d, M_Row_h * N_Col_h * sizeof(float));

        cudaMemcpy(M_d, M_h, M_Row_h * M_Col_h * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(N_d, N_h, N_Row_h * N_Col_h * sizeof(float), cudaMemcpyHostToDevice);


// === 2D GRID/BLOCK ===
    dim3 threadsPerBlock(2, 2);  // 2x2 = 4 threads per block
    dim3 blocksPerGrid(
        (N_Col_h + threadsPerBlock.x - 1) / threadsPerBlock.x,  // = (2+1)/2 = 2
        (M_Row_h + threadsPerBlock.y - 1) / threadsPerBlock.y   // = (3+1)/2 = 2
    );
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
        M_d, N_d, P_d, M_Row_h, N_Col_h, M_Col_h
    );



        cudaMemcpy(P_h, P_d, M_Row_h * N_Col_h * sizeof(float), cudaMemcpyDeviceToHost);
         // Verify a few elements
      for (int i = 0; i < M_Row_h * N_Col_h; ++i){
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
