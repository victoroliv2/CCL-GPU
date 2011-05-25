#include <stdio.h>

__global__ void global_coalesced_test_1(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int v = data[x]+1;
    data[x] = v;
}

__global__ void global_atomics_test_1(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    atomicAdd(&data[x], 1);
}

__global__ void local_coalesced_test_1(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int s_data[256];
    s_data[threadIdx.x] = data[x];
    __syncthreads();

    s_data[threadIdx.x]++;
    __syncthreads();

    data[x] = s_data[threadIdx.x];
}

__global__ void local_atomics_test_1(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int s_data[256];
    s_data[threadIdx.x] = data[x];
    __syncthreads();

    atomicAdd(&s_data[threadIdx.x], 1);
    __syncthreads();

    data[x] = s_data[threadIdx.x];
}

/////////////////////

__global__ void global_atomics_test_2(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int v = data[x];

    atomicAdd(&data[blockIdx.x], v);
}

__global__ void local_atomics_test_2(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int s_data[256];
    s_data[threadIdx.x] = data[x];
    __syncthreads();

    atomicAdd(&s_data[0], s_data[threadIdx.x]);
    __syncthreads();

    data[blockIdx.x] = s_data[0];
}

////////////////////////////////

__global__ void global_atomics_test_3(int* data) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int v = data[x];

    atomicAdd(&data[0], v);
}

////////////////////////////////


#define START_TIME cudaEventRecord(start,0)
#define STOP_TIME  cudaEventRecord(stop,0 ); \
                   cudaEventSynchronize(stop); \
                   cudaEventElapsedTime( &et, start, stop )

#define TRIES 10
float et_v[TRIES];

float MIN_ET() {
    float et = et_v[0]; 
    for (int t=0; t<TRIES; t++) { 
        et = (et_v[t] < et)? et_v[t] : et; 
    }
    return et;
}



int main(int argc, char* argv[]) {

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    float et;

    int N = 1<<20;
    int* data;
    cudaMalloc(&data, sizeof(int)*N);

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        global_coalesced_test_1 <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("global_coalesced_test_1: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        global_atomics_test_1   <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("global_atomics_test_1: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        local_coalesced_test_1  <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("local_coalesced_test_1: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        local_atomics_test_1    <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("locals_atomics_test_1: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        local_atomics_test_2    <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("global_atomics_test_2: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        local_atomics_test_2    <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("local_atomics_test_2: \t %lf\n", MIN_ET());

    for (int t=0;t<TRIES;t++) {
        START_TIME;
        global_atomics_test_3    <<<N/256, 256>>> (data);
        STOP_TIME;
        cudaThreadSynchronize();
        et_v[t] = et;
    }
    printf("local_atomics_test_3: \t %lf\n", MIN_ET());

    cudaFree(data);

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
