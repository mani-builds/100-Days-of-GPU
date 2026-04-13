// %%writefile matrix_vec_mul.cu
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

__global__ void MatVecMul(float *B_d, float *C_d, float *A_d, int B_row_d,
                          int Common_n_d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B_row_d) {
        float element = 0.0;
        for (int i = 0; i < Common_n_d; i++) {
            element += B_d[Common_n_d * row + i] * C_d[i];
        }
        A_d[row] = element;
    }
}

int main() {
    float *B_h, *C_h, *A_h;
    int B_row_h, Common_n_h;

    B_row_h = 3;
    Common_n_h = 2;
    B_h = (float *)malloc(B_row_h * Common_n_h * sizeof(float));
    C_h = (float *)malloc(Common_n_h * sizeof(float));
    A_h = (float *)malloc(B_row_h * sizeof(float));

    float B_temp[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float C_temp[] = {0.0, 2.0};

    memcpy(B_h, B_temp, B_row_h * Common_n_h * sizeof(float));
    memcpy(C_h, C_temp, Common_n_h * sizeof(float));

    float *B_d, *C_d, *A_d;
    int B_row_d, Common_n_d;

    B_row_d = B_row_h;
    Common_n_d = Common_n_h;

    cudaMalloc(&B_d, B_row_d * Common_n_d * sizeof(float));
    cudaMalloc(&C_d, Common_n_d * sizeof(float));
    cudaMalloc(&A_d, B_row_d * sizeof(float));

    cudaMemcpy(B_d, B_h, B_row_d * Common_n_d * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, Common_n_d * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (B_row_h + threads - 1) / threads;

    MatVecMul<<<blocks, threads>>>(B_d, C_d, A_d, B_row_d, Common_n_d);

    cudaMemcpy(A_h, A_d, B_row_d * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < B_row_h; i++) {
        printf("%f\n", A_h[i]);
    }

    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(A_d);
    free(B_h);
    free(C_h);
    free(A_h);

    return 0;
}
