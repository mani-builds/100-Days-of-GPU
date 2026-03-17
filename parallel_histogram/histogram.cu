#include <stdio.h>

__global__
void histogram_kernel(char *data, unsigned int length, unsigned int *histo) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if (i < length){
   int alphabet_pos = data[i] - 'a';
   if (alphabet_pos >= 0 && alphabet_pos < 26) {
     atomicAdd(&(histo[alphabet_pos/4]), 1);
     // histo[alphabet_pos/4]++;
   }
  }
}

int main() {
  char *data_h= "programming massively parallel processors";
  unsigned int total_len = 41;
  unsigned int *histo_h;
  histo_h = (unsigned int *)malloc(7*sizeof(unsigned int));

  char *data;
  unsigned int *histo;
  cudaMalloc(&histo,7 * sizeof(unsigned int));
  cudaMalloc(&data,total_len * sizeof(char));
  // cudaMemcpy(histo, histo_h, 7 * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(data, data_h, total_len * sizeof(char), cudaMemcpyHostToDevice);

  histogram_kernel<<<1, total_len>>>(data, total_len, histo);

  cudaMemcpy(histo_h, histo, 7*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("Printing bins: \n");
  for(int i=0; i<7; i++)
    printf("%d\t", histo_h[i]);
  printf("\n");

  // free(histo_h);
  // free(data_h);
  cudaFree(histo);
  cudaFree(data);

  return 0;
}
