#include <math.h>
#include <stdio.h>

#define TILE_WIDTH 2
__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and Col for the P element
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over M and N tiles required to compute P element
    float Pvalue = 0.0;
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) { // ph = Phase

        // Collabarative loading of M and N tiles into Mds and Nds
        Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();
    }
    P[Row * width + Col] = Pvalue;
}

int main() {
    // Assuming multiplication of square matrices
    int width_h;
    float *M_h, *N_h, *P_h;
    width_h = 4;

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
    dim3 blocksPerGrid((width_h + TILE_WIDTH - 1) / TILE_WIDTH,
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
