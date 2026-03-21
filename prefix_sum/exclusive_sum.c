#include <stdio.h>

int main() {

  int a[] = {3,1,7,0,4,1,6,3};
  int b[8];
  b[0] = 0;
  for (int i = 1; i < 8; i += 1) {
    b[i] = a[i-1] + b[i-1];
  }
  for (int i = 0; i < 8; i += 1) {
    /* b[i] = a[i-1] + b[i-1]; */
    printf("%d \t",b[i]);
  }
    printf("\n");

}
