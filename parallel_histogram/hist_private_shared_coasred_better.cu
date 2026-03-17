#include <stdio.h>

#define NUM_BINS 7
#define NUM_BLOCKS 3
#define CFACTOR 3

// Interleaved Coasring
//Best for GPU
__global__
void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {

   // Initialze privatized bins
   __shared__ unsigned int histo_s[NUM_BINS];
  // if NUM_BINS = size of block, then 'for' iterates for 1 iteration
   for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
   }
  __syncthreads();
  // Histogram
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid < length){
   for(int i=tid; i<  length; i+=blockDim.x * gridDim.x){
     // your data size should be higher than total threads in the grid
     //eg: you have 100 threads in your grid (blockDim.x * gridDim.x = 100) and 1,000 elements in your data
   int alphabet_pos = data[i] - 'a';
   if (alphabet_pos >= 0 && alphabet_pos < 26) {
     atomicAdd(&(histo_s[alphabet_pos/4]), 1);
   }
     }
   }
  __syncthreads();
     // Commit to Global Memory (HBW)
   for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
     unsigned int binValue = histo_s[bin];
     if(binValue > 0)
       atomicAdd(&(histo[bin]), histo_s[bin]);
     }
   }

int main() {
  char *data_h= "programming massively parallel processors";
  unsigned int total_len = 41;
  unsigned int *histo_h;
  histo_h = (unsigned int *)malloc(NUM_BINS*sizeof(unsigned int));

  char *data;
  unsigned int *histo;
  cudaMalloc(&histo,NUM_BLOCKS * NUM_BINS * sizeof(unsigned int));
  cudaMalloc(&data,total_len * sizeof(char));
  cudaMemcpy(histo, 0, NUM_BLOCKS * NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(data, data_h, total_len * sizeof(char), cudaMemcpyHostToDevice);

  dim3 blocksPerGrid = NUM_BLOCKS;
  dim3 threadPerBlock = ((total_len + blocksPerGrid.x - 1)/ blocksPerGrid.x);
  printf("Blocks: %d Threads: %d\n",  blocksPerGrid.x,threadPerBlock.x);
  histo_private_kernel<<<blocksPerGrid, threadPerBlock>>>(data, total_len, histo);

  cudaMemcpy(histo_h, histo, NUM_BLOCKS*NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("Printing bins: \n");
  for(int i=0; i<NUM_BINS; i++)
    printf("%d\t", histo_h[i]);
  printf("\n");

  // free(histo_h);
  // free(data_h);
  cudaFree(histo);
  cudaFree(data);

  return 0;
}
