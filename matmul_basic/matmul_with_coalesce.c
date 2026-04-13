#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//__global__ matMulCoalesceKernel() {}

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
    int count = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            B[i + j * width] = (float)count++;
        }
    }
    printf("Values of A: ");
    for (int k = 0; k < width * width; k++) {
        printf("%f\n", A[k]);
    }
    printf("Values of B: ");
    for (int k = 0; k < width * width; k++) {
        printf("%f\n", B[k]);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
