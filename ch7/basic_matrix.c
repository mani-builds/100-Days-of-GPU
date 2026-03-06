#include <stdio.h>
#include <stdlib.h>

int main() {
    /* Static multidimenional array (or matrix) */
    int A[4][4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int row = sizeof(A[0]) / sizeof(int);
    int col = sizeof(A) / sizeof(int) / row;
    printf("row: %d, col: %d\n", row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            printf("%d \t", A[i][j]);
        }

        printf("\n");
    }
    /* Dynamic multi-dimensional array */
    // Common way is to use a double pointer (One pointer to rows and each row
    // would have an array of pointers of length = col), it can be accessed via
    // mat[i][j]
    printf("----------------------------- Dyamic multi-dimensional array "
           "-----------------------------\n");
    int **mat = (int **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; i++) {
        mat[i] = (int *)malloc(col * sizeof(int));
    }
    int count = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            mat[i][j] = count++;
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d \t", mat[i][j]);
        }
        printf("\n");
    }

    // free the mem
    for (int i = 0; i < row; i++) {
        free(mat[i]);
    }
    free(mat);

    return 0;
}
