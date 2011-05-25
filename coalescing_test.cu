#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  

#define BLOCK_X 256

__global__ void
coalesced_read_1(float* f, int N) {
   int x = blockIdx.x*blockDim.x + threadIdx.x;
    
   float v;
   for (int i=0; i<100; i++) {
      v= f[x];
    }

   v *= v;
   f[x] = v;
}


__global__ void
coalesced_read_2(float* f, int N) {
   //int x = (gridDim.x-blockIdx.x-1)*blockDim.x + threadIdx.x;
   int x = blockIdx.x*blockDim.x + threadIdx.x;

   float v;

   for (int i=0; i<100; i++) {
      v = f[(x+1)%N];
    }

   v *= v;
   f[x] = v;
}


int main(int argc, char *argv[]) {

    int N = 1<<20;
    float* data;
    cudaMalloc(&data, sizeof(float)*N);

    coalesced_read_1 <<<N/256, 256>>> (data, N);
    cudaThreadSynchronize();

    coalesced_read_2 <<<N/256, 256>>> (data, N);
    cudaThreadSynchronize();

    cudaFree(data);

    return 0;
}
