nvcc -DTHREADED -arch sm_12 test.cu -o test
nvcc -Xptxas -v -arch sm_13 --maxrregcount=20 distance_transform2.cu -o distance_transform2
