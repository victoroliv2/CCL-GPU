#ifndef textures_cuh
#define textures_cuh

texture<unsigned char, 2, cudaReadModeElementType> imgtex;
texture<int, 1, cudaReadModeElementType> Ltex; 
texture<int, 1, cudaReadModeElementType> Rtex; 

#endif
