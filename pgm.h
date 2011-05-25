#include <stdio.h>
#include <stdlib.h>

typedef unsigned char uchar;

uchar* load_ppm(const char _imagefile[], int* _w, int* _h) {
  FILE* f;
  f = fopen(_imagefile, "r");

  uchar* image;

  int i,w,h;
  int g;
  char v[100];

  fgets(v, sizeof(v), f);
  //fgets(v, sizeof(v), f);
  fscanf(f, "%d %d", &w, &h);
  //printf("image size: %d %d\n", w, h);

  fgets(v, sizeof(v), f);
  fgets(v, sizeof(v), f);

  image = (uchar*)malloc(sizeof(uchar)*w*h);

  for(i=0; i<w*h; i++) {
    fscanf(f, "%d", &g);
    image[i] = g;
  }

  fclose(f);

  *_w = w;
  *_h = h;

  return image;
}



void save_ppm(const char _imagefile[], unsigned char* img, int w, int h) {
  FILE* f;
  f = fopen(_imagefile, "w");

  int i;

  fprintf(f, "P2\n");
  fprintf(f, "# CREATOR: GIMP PNM Filter Version 1.1\n");
  fprintf(f, "%d %d\n", w, h);
  fprintf(f, "255\n");

  for(i=0; i<w*h; i++) {
    fprintf(f, "%d\n", img[i]);
  }

  fclose(f);
}
