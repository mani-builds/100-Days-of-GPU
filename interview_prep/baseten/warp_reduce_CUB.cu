#include <stdio.h>
#include <cub/cub.cuh> // Using CUDA Unbound library

#define BLOCK_SIZE 512

__global__ void warp_reduce_kernel(float *in, float *out, int N) {
    // Specialize WarpReduce for type float
    typedef cub::WarpReduce<float> WarpReduce;

    // Allocate shared memory for ALL warps in the block.
    // CUB handles the indexing internally based on warp ID.
    __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_SIZE / 32];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0;

    // Load data
    if (i < N) sum += in[i];
    if (i + BLOCK_SIZE < N) sum += in[i + BLOCK_SIZE];

    // Determine which warp this thread belongs to
    int warpId = threadIdx.x / 32;

    // Compute the warp-wide reduction
    // Every thread in the warp participates, but only lane 0 returns the aggregate
    float aggregate = WarpReduce(temp_storage[warpId]).Sum(sum);

    // Every warp's leader (lane 0) adds its partial result to the global sum
    int laneId = threadIdx.x % 32;
    if (laneId == 0) {
        atomicAdd(out, aggregate);
    }
}

int main() {
    int N = 4096;
    float *in_h = (float *)malloc(N * sizeof(float));
    float *out_h = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; i++) in_h[i] = 1.0f;
    *out_h = 0.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, in_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int threads = BLOCK_SIZE;
    int blocksPerGrid = (N + (2 * threads) - 1) / (2 * threads);

    warp_reduce_kernel<<<blocksPerGrid, threads>>>(d_in, d_out, N);

    cudaMemcpy(out_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Blocks: %d, Threads: %d\n", blocksPerGrid, threads);
    printf("Output: %f \n", *out_h);

    free(in_h); free(out_h);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
