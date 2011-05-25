#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  

#include "pgm.h"

#define BLOCK_SIZE 128

__device__ float
distance(float x, float y, float ex, float ey) {
    return sqrtf(powf(ex-x,2.0f)+powf(ey-y,2.0f));
}

__global__ void
euclidian_distance_transform(uchar4* img, 
        float4* dist, int w, int h) {    

    //each thread process 4 pixels
   __shared__ uchar4 img_line [BLOCK_SIZE];

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float N = w*h;

    int4 ox = {(4*i) % w, (4*i+1) % w, (4*i+2) % w, (4*i+3) % w};
    int4 oy = {(4*i) / w, (4*i+1) / w, (4*i+2) / w, (4*i+3) / w};

    if (4*i < N) {
        float4 d = {N, N, N, N};
    
        for(int bi=0; bi<gridDim.x; bi++) {
            int nbi = (blockIdx.x+bi) % gridDim.x;
            int target = nbi*blockDim.x+threadIdx.x;

            //32-bit coalesced transaction
            img_line[threadIdx.x] = img[target];
            __syncthreads();

            for (int k=0;k<blockDim.x;k++) {
                float4 nd;
                //let's back to the original reference system
                int4 dx = {(4*target) % w, (4*target+1) % w, (4*target+2) % w, (4*target+3) % w};
                int4 dy = {(4*target) / w, (4*target+1) / w, (4*target+2) / w, (4*target+3) / w};

                //pixel 1 - x
                //pixel 2 - y

                nd.x = distance(float(ox.x),float(oy.x), float(dx.x),float(dy.x)); 
                nd.y = distance(float(ox.y),float(oy.y), float(dx.y),float(dy.y)); 
                nd.z = distance(float(ox.z),float(oy.z), float(dx.z),float(dy.z)); 
                nd.w = distance(float(ox.w),float(oy.w), float(dx.w),float(dy.w)); 

                uchar4 v = img_line[threadIdx.x];

                if (v.x == 1 && nd.x < d.x) d.x = nd.x;
                if (v.y == 1 && nd.y < d.y) d.y = nd.y;
                if (v.z == 1 && nd.z < d.z) d.z = nd.z;
                if (v.w == 1 && nd.w < d.w) d.w = nd.w;

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

    int w,h;
    unsigned char* img;
    img = load_ppm(argv[1], &w, &h);
    printf("%d %d\n", w, h);
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
    dim3 grid ((w*h+(4*BLOCK_SIZE-1))/(4*BLOCK_SIZE));
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    fprintf(stderr, "GO!");

    euclidian_distance_transform
        <<<grid, block>>> ((uchar4*)d_img, (float4*)d_dist, w, h);
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
