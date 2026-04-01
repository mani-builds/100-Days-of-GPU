#include <stdio.h>
#include <stdlib.h>

#define TILE 8
float* transpose(float *a, float *b, int M, int N) {

  float a_s[TILE][TILE];
  /* float b_s[TILE][TILE]; */

  for (int ph_row = 0; ph_row < M / TILE; ph_row++) {
    for (int ph_col = 0; ph_col < N / TILE; ph_col++) {
      // Block-wise loading
      for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < TILE; j++) {
          a_s[i][j] = a[(i + ph_row * TILE)* N + (j + ph_col * TILE)];
        }
      }
      for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < TILE; j++) {
          int trans_i = j + ph_col * TILE;
          int trans_j = i + ph_row * TILE;
          b[trans_i * M + trans_j] = a_s[i][j];
        }
      }
    }
  }
  return b;
}

int main() {
    int M = 32; // number of rows
    int N = 48; // number of columns

    size_t bytes_A = M * N * sizeof(float);
    size_t bytes_B = N * M * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);

    // Initialize input matrix
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            h_A[i*N + j] = i * N + j;

    printf("Original matrix (first 8x8 block):\n");
    for (int i = 0; i < 8 && i < M; ++i) {
        for (int j = 0; j < 8 && j < N; ++j)
            printf("%5.1f ", h_A[i*N + j]);
        printf("\n");
    }


    h_B = transpose(h_A, h_B, M, N);

    // Print part of the result for verification
    printf("Transposed matrix (first 8x8 block):\n");
    for (int i = 0; i < 8 && i < N; ++i) {
        for (int j = 0; j < 8 && j < M; ++j)
            printf("%5.1f ", h_B[i*M + j]);
        printf("\n");
    }

    free(h_A);
    free(h_B);

    return 0;
}
