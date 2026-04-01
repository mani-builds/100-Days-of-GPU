/* Reverse the bits of a 32-bit signed int */

#include <stdint.h>
#include <stdio.h>
#include <math.h>

int main() {

  int32_t a = 12816;
  /* int32_t a = 43261596; */
  int32_t b = 0;
  int rem;
  int array[32];
  for (int i=0; i<32; i++) array[i] = 0;
  int k = 0;
  if (a < 0) {
    array[0] = 1;
  }

  while (a !=0) {
    rem = a % 2;
    a = a / 2;
    array[31 - k] = rem;
    k++;
  }

  printf("Reverse Array \n");
  for (int i=0; i<32; i++) printf("%d",array[i]);
  printf("\n");

  //reverse
  k=0;
  for (int i = 0; i < 32; i++) {
    b += pow(2,k) * array[i];
    k++;
  }
   printf("\nb: %d \n", b);
}
