#include <stdio.h>
#include <stdlib.h>

namespace gold {

int find(int* label, int x) {
    if (label[x] == x) {
        return x;
    }else {
        int v = find(label, label[x]);
        label[x] = v;
        return v;
    }
}

void merge(int* label, int x, int y){
    x = find(label, x);
    y = find(label, y);

    if (label[x] < label[y]) {
        label[y] = x; 
    }else {
        label[x] = y;
    }
}

void CCL(unsigned char* img, int w, int h, int* label) {
    
    for (int i=0; i<w*h; i++) {
        label[i] = i;
    }

    for (int i=0; i<w*h; i++) {
        if (i%w>0  && img[i]==img[i-1]) merge(label, i, i-1);
        if (i/w>0  && img[i]==img[i-w]) merge(label, i, i-w);
    }
    for (int i=0; i<w*h; i++) {
        label[i] = find(label, i);
    }
}

}
