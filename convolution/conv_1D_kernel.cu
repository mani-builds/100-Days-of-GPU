#include <stdio.h>


__global__ void conv_1D_kernel(float *N, float *F, float *P, int n, int r){

  int idx = threadIdx.x  + blockIdx.x * blockDim.x;

  if (idx < n){
    int Pvalue = 0;
    for (int fIdx=0; fIdx<2*r+1; fIdx++){
      if(idx + fIdx - r >=0 && idx + fIdx - r < n){
        Pvalue += F[fIdx] * N[idx + fIdx- r];
      }
    }
    P[idx] = Pvalue;
  }


}

int main(){
  float N_h[] = {4,1,3,2,3};
  float F_h[] = {0.5,0.5,0.5};
  float *P_h;
  int n = sizeof(N_h) / sizeof(float);
  int r = sizeof(F_h) / sizeof(float) / 2;

  P_h=(float*)malloc(sizeof(float)*n);

  // dim3 threadsPerBlock(6);
  // dim3 blocksPerGrid((threadsPerBlock*n - 1 / n));


  float *N, *F, *P;
  cudaMalloc(&N, sizeof(float)*n);
  cudaMalloc(&P, sizeof(float)*n);
  cudaMalloc(&F, sizeof(float)*n + sizeof(int));

  cudaMemcpy(N, N_h, sizeof(float)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(F, F_h,sizeof(float)*n, cudaMemcpyHostToDevice);

  conv_1D_kernel<<<n, (n*n -1)/n>>>(N,F,P,n,r);

  cudaMemcpy(P_h, P, sizeof(float)*n, cudaMemcpyDeviceToHost);

  for(int i=0; i<n;i++){
    printf("%f\t", P_h[i]);
  }
    printf("\n");
  cudaFree(P);
  cudaFree(N);
  cudaFree(F);
  free(P_h);
}
