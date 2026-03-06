#include <stdio.h>
#include <stdlib.h>

int main() {
    int x_n, r;
    printf("Enter input length: \n");
    scanf("%d", &x_n);
    printf("Enter filter radius: \n");
    scanf("%d", &r);

    printf("%d, %d\n", x_n, r);

    /* int *x = (int *)malloc(x_n * sizeof(int)); */
    /* int *f = (int *)malloc(r * sizeof(int)); */

    int default_value = 0;
    int *y = (int *)malloc(x_n * sizeof(int));

    int N = 10; // set N for upper limit of rand()
    /* for (int i = 0; i < x_n; i++) { */
    /*     x[i] = rand() % (N + 1); */
    /* } */
    /* for (int i = 0; i < 2 * r + 1; i++) { */
    /*     if (i <= r) { */
    /*         f[i] = i + 1; */
    /*     } else if (i > r) { */
    /*         f[i] = 2 * r + 1 - i; */
    /*     } */
    /* } */
    int x[] = {8, 2, 5, 4, 1, 7, 3};
    int f[] = {1, 3, 5, 3, 1};
    int d;
    for (int i = 0; i < x_n; i++) {
        d = 0;
        for (int j = -r; j <= r; j++) {
            if (i + j < 0 || i + j >= x_n) {
                d += default_value;
            } else {
                d += f[r + j] * x[i + j];
            }
        }
        y[i] = d;
    }

    printf("filter f: \n");
    for (int i = 0; i < 2 * r + 1; i++) {
        printf("%d\n", f[i]);
    }
    printf("x : \n");
    for (int i = 0; i < x_n; i++) {
        printf("%d\n", x[i]);
    }

    printf("y: \n");
    for (int i = 0; i < x_n; i++) {
        printf("%d\n", y[i]);
    }
    return 0;
}
