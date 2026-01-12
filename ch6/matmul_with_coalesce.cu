#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 2
__global__ void matMulCoalesceKernel(float *A, float *B, float *C, int width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    printf("Row: %d, Col: %d\n", row, col);
    float Pvalue = 0.0;
    for (int ph = 0; ph < width / TILE_WIDTH; ph++) {

        if ((row < width) && (ph * TILE_WIDTH + tx) < width)
            Mds[ty][tx] = A[row * width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;
        // SWAP tx and ty for the B matrix global index calculation
        // This ensures that threads with consecutive tx access consecutive
        // memory in B
        if (((ph * TILE_WIDTH + tx) < width) && (bx * TILE_WIDTH + ty) < width)
            Nds[tx][ty] =
                B[(ph * TILE_WIDTH + tx) * width +
                  (bx * TILE_WIDTH + ty)]; // Note that B is a Column-major
                                           // matrix, so this access changes
        else
            Nds[tx][ty] = 0.0f;

        printf("ty: %d, tx: %d\n", ty, tx);
        printf("Mds: %f, Nds: %f\n", Mds[ty][tx], Nds[tx][ty]);
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += Mds[ty][i] * Nds[tx][i];
        }

        printf("Pvalue: %f\n", Pvalue);
        __syncthreads();
    }
    if ((row < width) && (col < width)) {

        C[row * width + col] = Pvalue;
    }
}

int main() {
    float *A, *B, *C; // A is a Row-major matrix and B is a Column-major matrix
    int width;

    width = 4;

    A = (float *)malloc(width * width * sizeof(float));
    B = (float *)malloc(width * width * sizeof(float));
    C = (float *)malloc(width * width * sizeof(float));

    // A is a Row-major matrix
    for (int i = 0; i < width * width; i++) {
        A[i] = (float)i;
    }
    // B is a Column-major matrix
    float count = 0.0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            B[i + j * width] = (float)count++;
        }
    }
    // printf("Values of A: ");
    // for (int k = 0; k < width * width; k++) {
    //     printf("%f\n", A[k]);
    // }
    // printf("Values of B: ");
    // for (int k = 0; k < width * width; k++) {
    //     printf("%f\n", B[k]);
    // }
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, width * width * sizeof(float));
    cudaMalloc(&B_d, width * width * sizeof(float));
    cudaMalloc(&C_d, width * width * sizeof(float));

    cudaMemcpy(A_d, A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + TILE_WIDTH - 1) / TILE_WIDTH,
                       (width + TILE_WIDTH - 1) / TILE_WIDTH);
    matMulCoalesceKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d,
                                                             width);
    cudaMemcpy(C, C_d, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Values of C: ");
    for (int k = 0; k < width * width; k++) {
        printf("%f\n", C[k]);
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);
    return 0;
}
