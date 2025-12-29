#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 2
#define COARSE_FACTOR 2
__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and Col for the P element
    int Row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++) {
        Pvalue[i] = 0.0f;
    }

    // Loop over M and N tiles required to compute P element
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) { // ph = Phase

        // Collabarative loading of M into Mds
        Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
        for (int c = 0; c < COARSE_FACTOR; ++c) {

            int col = colStart + TILE_WIDTH * c;
            // Collabarative loading of N into Nds
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) {
                Pvalue[c] += Mds[ty][i] * Nds[i][tx];
            }
            __syncthreads();
        }
    }
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + TILE_WIDTH * c;
        P[Row * width + col] = Pvalue[c];
    }
}

int main() {
    // Assuming multiplication of square matrices
    int width_h;
    float *M_h, *N_h, *P_h;
    width_h = 32;

    M_h = (float *)malloc(width_h * width_h * sizeof(float));
    N_h = (float *)malloc(width_h * width_h * sizeof(float));
    P_h = (float *)malloc(width_h * width_h * sizeof(float));

    for (int i = 0; i < width_h * width_h; i++) {
        M_h[i] = (float)i;
        N_h[i] = (float)i;
    }

    float *M_d, *N_d, *P_d;
    cudaMalloc(&M_d, width_h * width_h * sizeof(float));
    cudaMalloc(&N_d, width_h * width_h * sizeof(float));
    cudaMalloc(&P_d, width_h * width_h * sizeof(float));

    cudaMemcpy(M_d, M_h, width_h * width_h * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, width_h * width_h * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    // Notice the division by COARSE_FACTOR in the X dimension
    dim3 blocksPerGrid((width_h + (TILE_WIDTH * COARSE_FACTOR) - 1) /
                           (TILE_WIDTH * COARSE_FACTOR),
                       (width_h + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, P_d, width_h);
    cudaMemcpy(P_h, P_d, width_h * width_h * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < width_h * width_h; i++) {
        printf("%f\n", P_h[i]);
    }
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}
//
