#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  

#include "textures.cuh"

namespace naive_prop {

const int BLOCK_X = 16;
const int BLOCK_Y = 16;

__global__ void PROP_prescan(int* R, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;

    if (x < w && y < h) {
        R[index] = index;
    }
}

__global__ void PROP_scan(int* R, int w, int h, int* d_stop) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;

    if (x < w && y < h) {
        unsigned char v = tex2D(imgtex, x, y);
        int label = R[index];
        int newlabel = w*h;
 
        if (y>0 && tex2D(imgtex, x, y-1) == v) {
            newlabel = min(newlabel, R[index-w]);
        }
        if (y<h-1 && tex2D(imgtex, x, y+1) == v) {
            newlabel = min(newlabel, R[index+w]);
        }
        if (x>0 && tex2D(imgtex, x-1, y) == v) {
            newlabel = min(newlabel, R[index-1]);
        }
        if (x<w-1 && tex2D(imgtex, x+1, y) == v) {
            newlabel = min(newlabel, R[index+1]);
        }

        if (newlabel< label) {
            R[index] = newlabel;
            *d_stop = 0;
        }
    }
}

void CCL(unsigned char* img, int w, int h, int* label) {
    cudaError_t err;

    cudaArray* imgarray;
    cudaChannelFormatDesc uchardesc = 
        cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&imgarray, &uchardesc, w, h);

    int* R;
    cudaMalloc((void**)&R, w*h*sizeof(int));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("startERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaChannelFormatDesc intdesc = 
        cudaCreateChannelDesc<int>();
    cudaBindTextureToArray(imgtex, imgarray, uchardesc);
    cudaBindTexture(NULL, Rtex, R, intdesc, w*h*sizeof(int));

    int stop;
    int* d_stop;
    cudaMalloc((void**)&d_stop, sizeof(int));

    dim3 block (BLOCK_X, BLOCK_Y);
    dim3 grid ((w+BLOCK_X-1)/BLOCK_X,
               (h+BLOCK_Y-1)/BLOCK_Y);

    cudaMemcpyToArray(imgarray, 0, 0, img, 
            w*h*sizeof(unsigned char),
            cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("midERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    PROP_prescan <<<grid, block>>>
        (R, w, h);

    stop = 0;
    while (stop == 0) {

        cudaMemset(d_stop, 0xFF, sizeof(int));

        PROP_scan <<<grid, block>>>
            (R, w, h, d_stop);

        cudaMemcpy(&stop, d_stop, sizeof(int),
                cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(label, R, w*h*sizeof(int),
            cudaMemcpyDeviceToHost); 

    cudaFree(d_stop);
    cudaFree(R);
    cudaFreeArray(imgarray);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return;
    }
}

}
