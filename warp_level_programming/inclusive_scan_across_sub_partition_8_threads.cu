#include <stdio.h>

__global__ void inclusive_scan_8_threads_kernel() {
  int laneId = threadIdx.x % 32;
  int value = 31 - laneId; // Starting values to accumulate
    // Loop to accumulate scan within my partition.
    // Scan requires log2(8) == 3 steps for 8 threads
 for (int offset = 1; offset <=4; offset *= 2){
   int temp = __shfl_up_sync(0xFFFFFFFF, value, offset, /*width=*/ 8);
   int src_lane = laneId % 8 - offset;
   if (src_lane >= 0)
     value += temp;
   // The implicit lockstep execution of instructions in a warp
   // (due to SIMT nature of the warp) ensure that __shfl_down_sync is
   // executed by all the threads before moving on to the next instruction
 }
  printf("Thread: %d final value = %d \n", threadIdx.x, value);

}

int main() {
    inclusive_scan_8_threads_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
