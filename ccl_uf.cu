#include <stdio.h>  
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>  
#include <assert.h>
#include "textures.cuh"

#include <pthread.h>

namespace uf {

#define UF_BLOCK_SIZE_X 32
#define UF_BLOCK_SIZE_Y 16

class UnionFind
{
public:
    unsigned char* img;
    int* label;
    int width, height;
    int size;
    int divide;

    inline unsigned char F(int p) { return this->img[p]; }

    UnionFind(unsigned char* img, int* label, int w, int h, int divide);

    int find(int x);
    void merge(int x, int y);
    void build();
    void build(int ls, int le, int divide);
};

UnionFind::UnionFind(unsigned char* _img, int* _label, int _width, int _height, int _divide):
    width(_width), height(_height), size(_width*_height), divide(_divide), img(_img), label(_label)
{}

#ifdef THREADED
typedef struct _ThreadArg {
    UnionFind *uf;
    int ls;
    int le;
    int d;
} ThreadArg;

void BuildUnionFindThread(void *ptr)
{
    ThreadArg* arg = (ThreadArg* )ptr;
    UnionFind *uf = arg->uf;
    uf->build(arg->ls, arg->le, arg->d);
    pthread_exit(0);
}
#endif

void UnionFind::build()
{
    int nyblocks = (this->height+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y;
    this->build(0, nyblocks-1, this->divide);

    for(int i=0; i<this->width*this->height; i++)
        this->find(i);
}

void UnionFind::build(int ls, int le, int d)
{
    int rls = ls*UF_BLOCK_SIZE_Y;
    int rle = (le+1)*UF_BLOCK_SIZE_Y;
    if (rle > this->height) rle = this->height;

    //we have an image with labelled blocks of UF_BLOCK_SIZE_X x UF_BLOCK_SIZE_Y
    if (d == 0)
    {
        //fprintf(stderr, "%d\t%d\n", ls, le);

        //we have to merge blocks

        for(int y=rls; y<rle; y++)
        {
            //I imagine that if we merge horizontally first we get less cache misses
            //(untested)
            int offset = y*this->width;
            for(int x=UF_BLOCK_SIZE_X; x<this->width; x+=UF_BLOCK_SIZE_X)
            {
                this->merge(offset+x-1, offset+x);
            }
        }


        for(int y=rls+UF_BLOCK_SIZE_Y; y<rle; y+=UF_BLOCK_SIZE_Y)
        {
            int offset = y*this->width;
            for(int x=0; x<this->width; x++)
            {
                this->merge((offset-this->width)+x, offset+x);
            }
        }

    }
    else
    {
        int m = (ls+le)/2;
        #ifdef THREADED
        pthread_t thread1, thread2;
        ThreadArg arg1, arg2;
        arg1.uf      = this;
        arg2.uf      = this;
        arg1.ls      = ls;
        arg1.le      = m;
        arg1.d       = d-1;
        arg2.ls      = m+1;
        arg2.le      = le;
        arg2.d       = d-1;
        pthread_create(&thread1, NULL, (void *(*) (void *))&BuildUnionFindThread, (void *)&arg1);
        pthread_create(&thread2, NULL, (void *(*) (void *))&BuildUnionFindThread, (void *)&arg2);
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
        #else
        this->build(ls,  m,  d-1);
        this->build(m+1, le, d-1);
        #endif
        
        int b2 = ((m+1)*UF_BLOCK_SIZE_Y)*this->width;
        int b1 = b2+this->width;
        
        for(int x=0; x<this->width; x++)
        {
            this->merge(b1+x,b2+x);
        }

    }
}

int UnionFind::find(int x)
{
    int cur = x;
    int next = this->label[cur];
    while (next != cur)
    {
        cur = next;
        next = this->label[cur];
    }
    
    int root = next;

    cur = x;
    next = this->label[cur];
    while (next != cur)
    {
        this->label[cur] = root;
        cur = next;
        next = this->label[x];
    }
    
    return root;
}

void UnionFind::merge(int x, int y)
{
    if (F(x) == F(y)) 
    {
        x = this->find(x);
        y = this->find(y);

        assert(x == this->label[x]);
        assert(y == this->label[y]);

        if (x < y) {
            this->label[y] = x;
        } else {
            this->label[x] = y;
        }
    }
}

//CUDA

__device__ int find(int* buf, int x) {
    while (x != buf[x]) {
      x = buf[x];
    }
    return x;
}

__device__ void findAndUnion(int* buf, int g1, int g2) {
    bool done;    
    do {

      g1 = find(buf, g1);
      g2 = find(buf, g2);    
 
      // it should hold that g1 == buf[g1] and g2 == buf[g2] now
    
      if (g1 < g2) {
          int old = atomicMin(&buf[g2], g1);
          done = (old == g2);
          g2 = old;
      } else if (g2 < g1) {
          int old = atomicMin(&buf[g1], g2);
          done = (old == g1);
          g1 = old;
      } else {
          done = true;
      }

    } while(!done);
}

__global__ void UF_local(int* label, int w, int h) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    int block_index = UF_BLOCK_SIZE_X * threadIdx.y + threadIdx.x;

    __shared__ int s_buffer[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];
    __shared__ unsigned char s_img[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];

    bool in_limits = x < w && y < h;

    s_buffer[block_index] = block_index;
    s_img[block_index] = in_limits? tex2D(imgtex, x, y) : 0xFF;
    __syncthreads();

    unsigned char v = s_img[block_index];

    if (in_limits && threadIdx.x>0 && s_img[block_index-1] == v) {
        findAndUnion(s_buffer, block_index, block_index - 1);
    }

    __syncthreads();

    if (in_limits && threadIdx.y>0 && s_img[block_index-UF_BLOCK_SIZE_X] == v) {
        findAndUnion(s_buffer, block_index, block_index - UF_BLOCK_SIZE_X);
    }

    __syncthreads();

    if (in_limits) {
    int f = find(s_buffer, block_index);
    int fx = f % UF_BLOCK_SIZE_X;
    int fy = f / UF_BLOCK_SIZE_X;
    label[global_index] = (blockIdx.y*UF_BLOCK_SIZE_Y + fy)*w +
                            (blockIdx.x*blockDim.x + fx);
    }

}

__global__ void UF_global(int* label, int w, int h) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    
    bool in_limits = x < w && y < h;
    unsigned char v = (in_limits? tex2D(imgtex, x, y) : 0xFF);
 
    if (in_limits && y>0 && threadIdx.y==0 && tex2D(imgtex, x, y-1) == v) {
        findAndUnion(label, global_index, global_index - w);
    }

    if (in_limits && x>0 && threadIdx.x==0 && tex2D(imgtex, x-1, y) == v) {
        findAndUnion(label, global_index, global_index - 1);
    }

}


__global__ void UF_final(int* label, int w, int h) {    
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int global_index = x+y*w;
    
    bool in_limits = x < w && y < h;

    if (in_limits) {
        label[global_index] = find(label, global_index);
    }
}


void CCL(unsigned char* img, int w, int h, int* label, bool use_cpu=false) {
    cudaError_t err;
    cudaArray* imgarray;
    cudaChannelFormatDesc uchardesc = 
        cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&imgarray, &uchardesc, w, h);
    cudaBindTextureToArray(imgtex, imgarray, uchardesc);

    cudaMemcpyToArray(imgarray, 0, 0, img, 
            w*h*sizeof(unsigned char),
            cudaMemcpyHostToDevice);

    int* d_label;
    cudaMalloc((void**)&d_label, w*h*sizeof(int));

    dim3 block (UF_BLOCK_SIZE_X, UF_BLOCK_SIZE_Y);
    dim3 grid ((w+UF_BLOCK_SIZE_X-1)/UF_BLOCK_SIZE_X,
               (h+UF_BLOCK_SIZE_Y-1)/UF_BLOCK_SIZE_Y);
 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("startERROR: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

    UF_local <<<grid, block>>>
        (d_label, w, h);

    if (use_cpu)
    {
        cudaMemcpy(label, d_label, w*h*sizeof(int),
                cudaMemcpyDeviceToHost); 

        UnionFind m(img, label, w, h, 0);
        m.build();
    }
    else
    {
        cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

        UF_global <<<grid, block>>>
            (d_label, w, h);

        UF_final <<<grid, block>>>
            (d_label, w, h);

        cudaMemcpy(label, d_label, w*h*sizeof(int),
                cudaMemcpyDeviceToHost); 
    }

    cudaFree(d_label);
    cudaFreeArray(imgarray);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("endERROR: %s\n", cudaGetErrorString(err));
        return;
    }
}

}
