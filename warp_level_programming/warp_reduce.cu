#include <stdio.h>

__global__ void warp_reduce_kernel() {
  int laneId = threadIdx.x % 32;
  int value = 31 - laneId;

 for (int offset = 16; offset >=1; offset /= 2){
  value += __shfl_down_sync(0xFFFFFFFF, value, offset);
   // The implicit lockstep execution of instructions in a warp
   // (due to SIMT nature of the warp) ensure that __shfl_down_sync is
   // executed by all the threads before moving on to the next instruction
 }
  printf("Thread: %d final value = %d \n", threadIdx.x, value);

}

int main() {
    warp_reduce_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
