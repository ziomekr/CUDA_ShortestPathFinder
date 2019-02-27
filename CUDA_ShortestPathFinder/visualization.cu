#include "visualization.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#define INFINITY 2<<27
#define BLOCK_SIZE 512
#define GRID_SIZE 8192

//CUDA kernel for expansion of each field in matrix to square with edge = square_dimen_pixels
__global__
void expandMatrix(int* transformed, int* source, int n, int m, int square_dimen_pixels) {

	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	while (tIdx < n*m) {
		for (int i = 0; i < square_dimen_pixels; i++) {
			for (int j = 0; j < square_dimen_pixels; j++) {
				transformed[(tIdx / n)*n*square_dimen_pixels*square_dimen_pixels + (tIdx%n)*square_dimen_pixels + i * n*square_dimen_pixels + j] = source[tIdx];
			}
		}
		tIdx += blockDim.x*gridDim.x;
	}
}

//CUDA kernel assigning RGB values to each weight in matrix
__global__
void assignRGB(unsigned char* dest, int* src, int minVal, int maxVal, int pathMark, int blockedMark, int size) {
	#define c( x ) (255 * x)
	double granularity = 360.0 / ((double)(maxVal - minVal) + 1);
	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	while (tIdx < size) {
		unsigned char red, green, blue;
		if (src[tIdx] != pathMark && src[tIdx] != blockedMark) {
			double hue = (src[tIdx] - minVal) * granularity;
			int    H = (int)(hue / 60) % 6;
			double F = (hue / 60) - H;
			double Q = 1.0 - F;


			switch (H)
			{
			case 0:  red = c(1);  green = c(F);  blue = c(0);  break;
			case 1:  red = c(Q);  green = c(1);  blue = c(0);  break;
			case 2:  red = c(0);  green = c(1);  blue = c(F);  break;
			case 3:  red = c(0);  green = c(Q);  blue = c(1);  break;
			case 4:  red = c(F);  green = c(0);  blue = c(1);  break;
			default: red = c(1);  green = c(0);  blue = c(Q);
			}

		}
		else {
			if (src[tIdx] == blockedMark) {
				blue = green = red = c(0);
			}
			else {
				blue = green = red = c(0.5);
			}

		}
		dest[tIdx * 3] = blue;
		dest[tIdx*3 + 1] = green;
		dest[tIdx * 3 + 2] = red;
		tIdx += blockDim.x*gridDim.x;
	}
	#undef c

}


struct writeToStream
{
	unsigned long value;
	unsigned      size;
	writeToStream(unsigned long value, unsigned size) :
		value(value), size(size)
	{ }
};

inline std::ostream& operator << (std::ostream& outs, const writeToStream& v)
{
	unsigned long value = v.value;
	for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
		outs.put(static_cast <char> (value & 0xFF));
	return outs;
}


bool makeBMP(const std::string& filename, unsigned char* RGBMatrix, int rows, int columns) {
	std::ofstream f(filename.c_str(),
		std::ios::out | std::ios::trunc | std::ios::binary);
	if (!f) return false;

	unsigned long headers_size = 14 + 40; 
	unsigned long pixel_data_size = rows * columns*3;

	// Write the BITMAPFILEHEADER
	f.put('B').put('M');                          
	f << writeToStream(headers_size + pixel_data_size, 4); 
	f << writeToStream(0, 2);
	f << writeToStream(0, 2); 
	f << writeToStream(headers_size, 4);  

	// Write the BITMAPINFOHEADER
	f << writeToStream(40, 4);  
	f << writeToStream(columns, 4);  
	f << writeToStream(rows, 4);  
	f << writeToStream(1, 2);  
	f << writeToStream(24, 2);  
	f << writeToStream(0, 4);  
	f << writeToStream(pixel_data_size, 4);  
	f << writeToStream(0, 4);  
	f << writeToStream(0, 4);  
	f << writeToStream(0, 4);  
	f << writeToStream(0, 4); 
	// Write RGB matrix to stream
	for (unsigned long i = 0; i < rows*columns*3; i++) {
		f.put(static_cast <char> (RGBMatrix[i]));
	}
	return f.good();
}

//Adding found path to matrix
void addPathToMatrix(int* matrix, int* path, int size) {
	int idx = size - 1;
	while (idx > 0) {
		matrix[idx] = -1;
		idx = path[idx];
	}
	matrix[0] = -1;
}

//Wrapping function producing colorful bitmap out of the matrix
void visualizeMatrix(const std::string& filename, int* matrix, int n, int m, int pixel_dimension, int minWeight, int maxWeight) {
	
	int *mGPU, *tGPU;
	unsigned char *rgb, *rgbGPU;

	//Memory allocation
	int* transformed = (int*)malloc(n*m * sizeof(int) * pixel_dimension * pixel_dimension);
	rgb = (unsigned char*)malloc(sizeof(unsigned char)*n*m*pixel_dimension*pixel_dimension * 3);
	cudaMalloc(&rgbGPU, sizeof(unsigned char)*n*m*pixel_dimension*pixel_dimension * 3);
	cudaMalloc(&tGPU, sizeof(int)*n*m*pixel_dimension*pixel_dimension);
	cudaMalloc(&mGPU, sizeof(int)*n*m);
	cudaMemcpy(mGPU, matrix, sizeof(int)*n*m, cudaMemcpyHostToDevice);

	//Actual work
	expandMatrix << <GRID_SIZE, BLOCK_SIZE >> > (tGPU, mGPU, n, m, pixel_dimension);
	assignRGB << <GRID_SIZE, BLOCK_SIZE >> > (rgbGPU, tGPU, minWeight, maxWeight, -1, INFINITY, n*m*pixel_dimension*pixel_dimension);
	cudaMemcpy(rgb, rgbGPU, sizeof(unsigned char)*n*m*pixel_dimension*pixel_dimension * 3, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	makeBMP(filename, rgb, n*pixel_dimension, m*pixel_dimension);

	//Memory deallocation
	free(transformed);
	free(rgb);
	cudaFree(rgbGPU);
	cudaFree(tGPU);
	cudaFree(mGPU);
}
#undef GRID_SIZE
#undef BLOCK_SIZE
