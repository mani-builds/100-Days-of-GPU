#include <stdio.h>

#define SECTION_SIZE 16
#define COARSE_FACTOR 4
#define n 128
__global__ void Kogge_Stone_Coarse_scan_kernel(float *x, float *y, int N) {
    // We partition the block’s section of the input into multiple contiguous
    // subsections: one for each thread. The number of subsections is the same as the
    // number of threads in the thread block
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float XY[SECTION_SIZE];

  // Iteratively load from Global Mem into shared memory (Coalesced)
  for (int j = threadIdx.x; j < SECTION_SIZE; j +=blockDim.x) {
    XY[j] = x[j];
  }
  __syncthreads(); // Make sure all the data is loaded onto SM

  // Phase 1
  // Thread 0 performs scan on a contiguous 'section' of XY and Thread 1 the next
  // and so on
  for (int index = threadIdx.x * COARSE_FACTOR + 1;
       index < (threadIdx.x + 1)*COARSE_FACTOR; index += 1) {
    // Start index +1 coz, we don't want to add our initial value of section
    if (index < SECTION_SIZE ) {
      XY[index] = XY[index] + XY[index - 1];
    }
  }
  __syncthreads(); // Make sure Phase 1 is done

  // Phase 2
  // Apply Kogge-Stone or Brent-Kung algo on Phase 1 results, but only on the
  // logical array that contains last elements of each section
  // Each iteration peforms one level of additions on 'logical array', and
  // no. of additions decrease over iteration (Ref: Fig 11.2)
  int num_element = SECTION_SIZE / COARSE_FACTOR; // No. of elements in logical array
  for (int offset = 1; offset < num_element; offset *= 2) {
    float temp;
    int last_idx = (threadIdx.x + 1) * COARSE_FACTOR - 1;
    if (threadIdx.x >= offset) {
      int src_idx = last_idx - offset * COARSE_FACTOR;
      if (last_idx < SECTION_SIZE && src_idx >= 0) {
        temp = XY[last_idx] + XY[src_idx];
      }
    }
    __syncthreads(); // To prevent Read-after-write race condition
   if (threadIdx.x >= offset && last_idx < SECTION_SIZE){
      XY[last_idx] = temp;
    }
    __syncthreads();
  }

  // Phase 3
  // Each threads adds the new value of the last element of predecessor's section
  // to its element
  int last_element_idx = (threadIdx.x + 1) * COARSE_FACTOR - 1;
  for (int k = (threadIdx.x + 1) * COARSE_FACTOR;
       k < (threadIdx.x + 1) * COARSE_FACTOR + COARSE_FACTOR - 1; k++) {
    // Start idx from start of next section and end before the end of later section
    if(k < SECTION_SIZE) XY[k] = XY[k] + XY[last_element_idx];
  }

  // Write it to Global Memory
  // Each thread writes the same no. of elements it moved to SM
  for (int j = threadIdx.x; j < SECTION_SIZE; j +=blockDim.x) {
    y[j] = XY[j];
  }
}

int main() {

  // float *x_h;
  float *y_h;
  int N;

  N = 16;
  // x_h = (float *) malloc(N*sizeof(float));
  y_h = (float *) malloc(N*sizeof(float));

  // for (int i=0;i <N; i++) x_h[i] = 1;
  float x_h[] = {2,1,3,1,0,4,1,2,0,3,1,2,5,3,1,2};

  printf("\nInput \n");
  for (int i=0;i <N; i++) printf("%f \t ",x_h[i]);
  float *x;
  float *y;

  cudaMalloc(&x, N*sizeof(float));
  cudaMalloc(&y, N*sizeof(float));

  cudaMemcpy(x, x_h, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(y, y_h, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = SECTION_SIZE/COARSE_FACTOR;
  int blocksPerGrid = 1;
  printf("\nGridDIM: %d, BlockDIM: %d\n", blocksPerGrid, threads);
  Kogge_Stone_Coarse_scan_kernel<<<blocksPerGrid, threads>>>(x, y, N);

  cudaMemcpy(y_h, y, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nOutput \n");
  for (int i=0;i <N; i++) printf("%f \t ",y_h[i]);
  printf("\n");

  // free(x_h);
  free(y_h);
  cudaFree(x);
  cudaFree(y);

  return 0;
}
