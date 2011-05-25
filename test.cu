#include <stdio.h>
#include <cuda.h>

#include "pgm.h"

#include "ccl_gold.cpp"
#include "ccl_uf.cu"
#include "ccl_lequiv.cu"
#include "ccl_naive_prop.cu"


int number_cc(int* label, int w, int h) {
    bool* mask = (bool*)malloc(w*h*sizeof(bool));
    for (int i=0; i<w*h; i++) {
        mask[i] = (label[i] == i);
    }
    int count = 0;
    for (int i=0; i<w*h; i++) {
        if (mask[i]) {count++;}
    }
    free(mask);
    return count;
}


#define TRIES 1
float et_v[TRIES];

#define START_TIME cudaEventRecord(start,0)
#define STOP_TIME  cudaEventRecord(stop,0 ); \
                   cudaEventSynchronize(stop); \
                   cudaEventElapsedTime( &et, start, stop )

int w,h;
unsigned char* img;
int *label, *label_gold;

void VERIFY() {
    for (int k=0;k<w*h;k++) { 
        if (label[k] != label_gold[k]) { 
            printf("WRONG!\n"); 
            break; 
        } 
    } 
    for (int k=0;k<w*h;k++) { 
        label[k] = -1; 
    } 
}

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

        //printf("======================================\n");
        //printf("%s\n", argv[1]);
        img = load_ppm(argv[1], &w, &h);
        label_gold = (int*)malloc(w*h*sizeof(int));
        label = (int*)malloc(w*h*sizeof(int));

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            gold::CCL(img, w, h, label_gold);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        printf("cc: %d\n", number_cc(label_gold, w, h));
        printf("gold: %.3f\n", et);        

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            uf::CCL(img, w, h, label, false);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        
        printf("uf_gpu_total: %.3f\n", et);        

        VERIFY();

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            uf::CCL(img, w, h, label, true);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        
        printf("uf_cpu+gpu_total: %.3f\n", et);        

        VERIFY();

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            lequiv::CCL(img, w, h, label);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();

        printf("lequiv_total: %.3f\n", et);        

        VERIFY();

        /*for (int t=0;t<TRIES;t++) {
            START_TIME;
            naive_prop::CCL(img, w, h, label);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();

        printf("%.3f %.3f\n", et, et_gold/et);        

        VERIFY();*/

        free(img);
        free(label);

    return 0;
}
