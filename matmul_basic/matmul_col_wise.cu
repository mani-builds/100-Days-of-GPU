#include <math.h>
#include <stdio.h>

__global__ void MatMulKernelCol(float *M_d, float *N_d, float *P_d, int M_Row_d,
                                int N_Col_d, int CommonDim_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N_Col_d) {
        for (int j = 0; j < M_Row_d; j++) {
            float element_value = 0.0;
            for (int i = 0; i < CommonDim_d; i++) {
                element_value +=
                    M_d[j * CommonDim_d + i] * N_d[N_Col_d * i + col];
            }
            P_d[j * N_Col_d + col] = element_value;
        }
    }
}

int main() {
    float *M_h, *N_h, *P_h;
    int M_Row_h, N_Col_h, M_Col_h, N_Row_h;

    M_Row_h = 3;
    M_Col_h = 2;
    N_Row_h = 2;
    N_Col_h = 3;

    M_h = (float *)malloc(M_Row_h * M_Col_h * sizeof(float));
    N_h = (float *)malloc(N_Row_h * N_Col_h * sizeof(float));
    P_h = (float *)malloc(M_Row_h * N_Col_h * sizeof(float));

    float M_temp[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float N_temp[] = {0.0, 2.0, 1.0, 1.0, 0.0, 3.0};
    memcpy(M_h, M_temp, M_Row_h * M_Col_h * sizeof(float));
    memcpy(N_h, N_temp, N_Row_h * N_Col_h * sizeof(float));

    if (M_Col_h == N_Row_h) {
        float *M_d, *N_d, *P_d;
        int M_Row_d, N_Col_d, CommonDim_d;

        M_Row_d = M_Row_h;
        N_Col_d = N_Col_h;
        CommonDim_d = N_Row_h;

        cudaMalloc(&M_d, M_Row_h * M_Col_h * sizeof(float));
        cudaMalloc(&N_d, N_Row_h * N_Col_h * sizeof(float));
        cudaMalloc(&P_d, M_Row_h * N_Col_h * sizeof(float));

        cudaMemcpy(M_d, M_h, M_Row_h * M_Col_h * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(N_d, N_h, N_Row_h * N_Col_h * sizeof(float),
                   cudaMemcpyHostToDevice);

        // === 1D GRID/BLOCK ===
        int threadsPerBlock = 128;
        int blocksPerGrid = (M_Row_d + threadsPerBlock - 1) / threadsPerBlock;
        MatMulKernelRow<<<blocksPerGrid, threadsPerBlock>>>(
            M_d, N_d, P_d, M_Row_d, N_Col_d, CommonDim_d);

        cudaMemcpy(P_h, P_d, M_Row_d * N_Col_d * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(M_d);
        cudaFree(N_d);

    } else {
        printf("Dimensions do not match, so can't multiply.\n");
    }

    // Verify a few elements
    for (int i = 0; i < M_Row_h * N_Col_h; ++i) {
        printf("%f\n", P_h[i]);
    }

    free(M_h);
    free(N_h);
    free(P_h);
}
