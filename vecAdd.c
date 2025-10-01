#include <stdio.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n){
  for(int i = 0; i<n; i++){
    C_h[i] = A_h[i] + B_h[i];
  }
}

int main(int argc, char *argv[])
{
  int n = 5;
  float A[n];
  float B[n];
  float C[n];
 
  for(int i = 0; i<n; i ++){
    A[i] = (float) i + 1;
    B[i] =  (float) 10 + i;
  }
  printf("Array A's address: %p \n", A);
  printf("Array A: %f \n", *A);
  printf("Array A: %f \n", A[0]);
  vecAdd(A, B, C, n);
  return 0;
}
