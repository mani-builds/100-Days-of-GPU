#include <cassert>
#include <stdio.h>

__global__ void warp_broadcast_kernel(int input) {
  int value;
  int laneId = threadIdx.x % 32;

  if (laneId == 0){
    value = input; // set the value only for threadIdx = 0
  }
  value = __shfl_sync(0xFFFFFFFF, value, 0); // use that value from thread 0
                                             // to set for all the others in that warp
                                             //Synchronize all threads in warp,
                                             //and get "value" from lane 0
  assert(value == input);
}

int main() {
  int input = 1204;

  warp_broadcast_kernel<<<1, 32>>>(input);
  printf("Kernel executed.\n");
  return 0;
}
