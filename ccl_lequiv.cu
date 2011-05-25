#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  

#include "textures.cuh"

namespace lequiv {

#define LEQUIV_BLOCK_SIZE_X 16
#define LEQUIV_BLOCK_SIZE_Y 16


__global__ void LEQUIV_prescan(int* L, int* R, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;

    if (x < w && y < h) {
        L[index] = index;
        R[index] = index;
    }
}

__global__ void LEQUIV_scan(int* R, int w, int h, int* d_stop) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;

    if (x < w && y < h) {
        unsigned char v = tex2D(imgtex, x, y);
        int label = tex1Dfetch(Ltex, index);
        int newlabel = w*h;
 
        if (y>0 && tex2D(imgtex, x, y-1) == v) {
            newlabel = min(newlabel, tex1Dfetch(Ltex, index-w));
        }
        if (y<h-1 && tex2D(imgtex, x, y+1) == v) {
            newlabel = min(newlabel, tex1Dfetch(Ltex, index+w));
        }
        if (x>0 && tex2D(imgtex, x-1, y) == v) {
            newlabel = min(newlabel, tex1Dfetch(Ltex, index-1));
        }
        if (x<w-1 && tex2D(imgtex, x+1, y) == v) {
            newlabel = min(newlabel, tex1Dfetch(Ltex, index+1));
        }

        if (newlabel< label) {
            R[label] = newlabel;
            *d_stop = 0;
        }
    }
}

__global__ void LEQUIV_analysis(int* L, int* R, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;
    int label;

    if (x < w && y < h) {
        label = L[index];
        if (label == index) {
            int deep = 128;
            int rf = label;
            //label = tex1Dfetch(Rtex, rf);
            label = R[rf];
            while (rf!=label && deep>0) {
                rf = label;
                label = tex1Dfetch(Rtex, rf);
                deep--;
            }
            //texture will be invalid
            R[index] = label;
        }
    }
}

__global__ void LEQUIV_labeling(int* L, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x+y*w;
    if (x < w && y < h) {
        int label = L[index];
        int cc = tex1Dfetch(Rtex, label);
        L[index] = tex1Dfetch(Rtex, cc);
    }
}

void CCL(unsigned char* img, int w, int h, int* label) {
    cudaError_t err;

    cudaArray* imgarray;
    cudaChannelFormatDesc uchardesc = 
        cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&imgarray, &uchardesc, w, h);

    int* L;
    cudaMalloc((void**)&L, w*h*sizeof(int));
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
    cudaBindTexture(NULL, Ltex, L, intdesc, w*h*sizeof(int));
    cudaBindTexture(NULL, Rtex, R, intdesc, w*h*sizeof(int));

    int stop;
    int* d_stop;
    cudaMalloc((void**)&d_stop, sizeof(int));

    dim3 block (LEQUIV_BLOCK_SIZE_X, LEQUIV_BLOCK_SIZE_Y);
    dim3 grid ((w+LEQUIV_BLOCK_SIZE_X-1)/LEQUIV_BLOCK_SIZE_X,
               (h+LEQUIV_BLOCK_SIZE_Y-1)/LEQUIV_BLOCK_SIZE_Y);

    cudaMemcpyToArray(imgarray, 0, 0, img, 
            w*h*sizeof(unsigned char),
            cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("midERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    LEQUIV_prescan <<<grid, block>>>
        (L, R, w, h);

    stop = 0;
    while (stop == 0) {

        cudaMemset(d_stop, 0xFF, sizeof(int));

        LEQUIV_scan <<<grid, block>>>
            (R, w, h, d_stop);

        LEQUIV_analysis <<<grid, block>>>
            (L, R, w, h);

        LEQUIV_labeling <<<grid, block>>>
            (L, w, h);

        cudaMemcpy(&stop, d_stop, sizeof(int),
                cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(label, L, w*h*sizeof(int),
            cudaMemcpyDeviceToHost); 

    cudaFree(d_stop);
    cudaFree(L);
    cudaFree(R);
    cudaFreeArray(imgarray);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return;
    }
}

}
