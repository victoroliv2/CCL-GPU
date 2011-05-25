#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  

#include "pgm.h"

#define BLOCK_SIZE 256

__device__ float
distance(float x, float y, float ex, float ey) {
    return sqrtf(powf(ex-x,2.0f)+powf(ey-y,2.0f));
}

__global__ void
euclidian_distance_transform(uchar2* img, 
        float2* dist, int w, int h) {    

    //each thread process 4 pixels
   __shared__ uchar2 img_line [BLOCK_SIZE];

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float N = w*h;

    int2 ox = {(2*i) % w, (2*i+1) % w};
    int2 oy = {(2*i) / w, (2*i+1) / w};

    if (2*i < N) {
        float2 d = {N, N};
    
        for(int bi=0; bi<gridDim.x; bi++) {
            int nbi = (blockIdx.x+bi) % gridDim.x;
            int target = nbi*blockDim.x+threadIdx.x;

            //32-bit coalesced transaction
            img_line[threadIdx.x] = img[target];
            __syncthreads();

            for (int k=0;k<blockDim.x;k++) {
                float2 nd;
                //let's back to the original reference system
                int2 dx = {(2*target) % w, (2*target+1) % w};
                int2 dy = {(2*target) / w, (2*target+1) / w};

                //pixel 1 - x
                //pixel 2 - y

                nd.x = distance(float(ox.x),float(oy.x), float(dx.x),float(dy.x)); 
                nd.y = distance(float(ox.y),float(oy.y), float(dx.y),float(dy.y)); 

                uchar2 v = img_line[threadIdx.x];

                if (v.x == 1 && nd.x < d.x) d.x = nd.x;
                if (v.y == 1 && nd.y < d.y) d.y = nd.y;

            }
        }
    
        //128-bit coalesced transaction
        dist[i] = d;

    }
}

int main(int argc, char* argv[]) {
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    //float et;
    cudaError_t err;

    unsigned char *img;
    img = load_ppm(argv[1], &w, &h);
    unsigned char *d_img;
    cudaMalloc((void**) &d_img, w*h*sizeof(unsigned char));
    cudaMemcpy(d_img, img, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    fprintf(stderr, "GO!");

    //-------------------
 
    float* dist = (float*)malloc(w*h*sizeof(float));
    float* d_dist;
    cudaMalloc((void**) &d_dist, w*h*sizeof(float));

    dim3 block (BLOCK_SIZE,1);
    dim3 grid ((w*h+(2*BLOCK_SIZE-1))/(2*BLOCK_SIZE));
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    fprintf(stderr, "GO!");

    euclidian_distance_transform
        <<<grid, block>>> ((uchar2*)d_img, (float2*)d_dist, w, h);
    cudaThreadSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    fprintf(stderr, "GO!");

    cudaMemcpy(dist, d_dist, w*h*sizeof(float), cudaMemcpyDeviceToHost);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_img);
    cudaFree(d_dist);
    free(dist);
    free(img);

    return 0;
}
