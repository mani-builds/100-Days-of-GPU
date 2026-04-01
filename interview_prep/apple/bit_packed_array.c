/* Store four 4-bit integers inside a single uint16_t
 * and write a function to retrieve the n-th integer. */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

int n_integer(uint16_t a, int n) {
  if (n > 3) return -1;
  int k = 0;
  int value=0,rem;
  int array[16];
  for (int i=0; i<16; i++) array[i] = 0;
  while (a !=0) {
    rem = a % 2;
    a = a / 2;
    array[15-k] = rem;
    k++;
  }
  k = 0;
  for (int i = 4*n; i < 4*(n + 1); i++) {
    value += pow(2,k) * array[15-i];
    k++;
  }
  return value;

}

int main() {
  uint16_t a=0;

  bool int_0[] = {0,0,0,0};
  bool int_1[]= {0,0,0,1};
  bool int_2[]= {0,0,1,0};
  bool int_3[]= {0,0,1,1};

  bool array[16] ;
  for (int i = 0; i < 16; i++) {
    if (i < 4)
    array[i] = int_3[i];
    else if (i >=4 && i < 8)
    array[i] = int_2[i-4];
    else if (i >=8 && i < 12)
    array[i] = int_1[i-8];
    else if (i >=12 && i < 16)
    array[i] = int_0[i-12];
  }

  for (int i = 0; i < 16; i++) printf("%d", array[i]);
  int k=0;
  for (int i = 0; i < 16 ; i++) {
    a += pow(2,k) * array[15-i];
    k++;
  }
  printf("\na: %d\n", a);
  int n = 3;
  printf("\n%d th integer: %d \n", n, n_integer(a,n));
}
