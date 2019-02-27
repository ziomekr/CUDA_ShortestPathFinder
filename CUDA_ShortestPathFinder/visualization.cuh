#pragma once
#include<string>
__global__ void expandMatrix(int* transformed, int* source, int n, int m, int square_dimen_pixels);
__global__ void assignRGB(unsigned char* dest, int* src, int minVal, int maxVal, int pathMark, int blockedMark, int size);
bool makeBMP(const std::string& filename, unsigned char* RGBMatrix, int rows, int columns);
void addPathToMatrix(int* matrix, int* path, int size);
void visualizeMatrix(const std::string& filename, int* matrix, int n, int m, int pixel_dimension, int minWeight, int maxWeight);