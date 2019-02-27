//GPU version of Dijsktra algorithm

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DijkstraGPU.cuh"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <stdio.h>


#define BLOCK_SIZE 512
#define GRID_SIZE 1024*16
#define INFINITY 2<<27

//Setting up needed arrays
__global__ 
void prepareArrays(int* costs, int* updatingCosts, bool* masks, int* path, int size) {
	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;;
	while (tIdx < size) {
		costs[tIdx] = updatingCosts[tIdx] = INFINITY;
		masks[tIdx] = false;
		path[tIdx] = -1;
		tIdx += blockDim.x*gridDim.x;
	}
	masks[0] = true;
	costs[0] = updatingCosts[0] = 0;
}

//Visiting left and right field and updating cost if lower
__device__ 
void visitHorizontalNeighbor(int tIdx, int* matrix, int* cost, int* updatingCost, int* path, int n, int matrix_size) {
	int neighbour_idx;
	for (int i = -1; i < 2; i += 2) {
		neighbour_idx = tIdx + i;
		if ((neighbour_idx / n == tIdx / n) && (neighbour_idx > -1) && (neighbour_idx < matrix_size)) {
			if (updatingCost[neighbour_idx] > cost[tIdx] + matrix[neighbour_idx]) {
				updatingCost[neighbour_idx] = cost[tIdx] + matrix[neighbour_idx];
				path[neighbour_idx] = tIdx;
			}
		}
	}
}
//Visiting up and down field and updating cost if lower
__device__
void visitVerticalNeighbor(int tIdx, int* matrix, int* cost, int* updatingCost, int* path, int n, int matrix_size) {
	
	int neighbour_idx;
	for (int i = -1; i < 2; i += 2) {
		neighbour_idx = tIdx + n*i;
		if ((neighbour_idx > -1) && (neighbour_idx < matrix_size)) {
			if (updatingCost[neighbour_idx] > cost[tIdx] + matrix[neighbour_idx]) {
				updatingCost[neighbour_idx] = cost[tIdx] + matrix[neighbour_idx];
				path[neighbour_idx] = tIdx;
			}
		}
	}
}

//CUDA kernel visiting all neigboring fields if mask set to true
__global__
void visitNeighbors(int* matrix, bool* masks, int* cost, int* updatingCost, int* path, int n, int matrix_size){
	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	while (tIdx < matrix_size) {
		if (masks[tIdx]) {
			masks[tIdx] = false;
			visitVerticalNeighbor(tIdx, matrix, cost, updatingCost, path, n, matrix_size);
			visitHorizontalNeighbor(tIdx, matrix, cost, updatingCost, path, n, matrix_size);
		}
		tIdx += blockDim.x*gridDim.x;
	}
}

//CUDA kernel for cost update if new cost was lower, RELAXATION kernel
__global__
void updateCosts(int* costs, int* updatingCosts, bool* masks, int matrix_size, bool *wasUpdated) {
	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	while (tIdx < matrix_size) {
		if (costs[tIdx] > updatingCosts[tIdx]) {
			costs[tIdx] = updatingCosts[tIdx];
			masks[tIdx] = true;
			*wasUpdated = true;
		}
		updatingCosts[tIdx] = costs[tIdx];
		tIdx += blockDim.x*gridDim.x;
	}
}

//
int* DijkstraGPU(int* matrix, int n, int m) {

	//Initialization and memory allocation
	int matrix_size = n * m;
	int *matrixGPU, *costsGPU, *updatingCostsGPU, *pathGPU;
	bool* masksGPU, *masks;
	bool *doWork, *doWorkGPU;
	int* path = (int*)malloc(matrix_size * sizeof(int));
	masks = (bool*)malloc(matrix_size * sizeof(bool));
	cudaMalloc(&matrixGPU, sizeof(int)*matrix_size);
	cudaMalloc(&costsGPU, sizeof(int)*matrix_size);
	cudaMalloc(&updatingCostsGPU, sizeof(int)*matrix_size);
	cudaMalloc(&pathGPU, sizeof(int)*matrix_size);
	cudaMalloc(&masksGPU, sizeof(bool)*matrix_size);
	cudaMemcpy(matrixGPU, matrix, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

	
	prepareArrays <<<GRID_SIZE, BLOCK_SIZE >>> (costsGPU, updatingCostsGPU, masksGPU, pathGPU, matrix_size);
	
	//Locked page pointer so the global flag can be read faster
	cudaHostAlloc(&doWork, sizeof(bool), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&doWorkGPU, doWork, 0);
	*doWork = true;

	//Doing work while updates occur
	while (*doWork) {
		*doWork = false;
		visitNeighbors <<<GRID_SIZE, BLOCK_SIZE >>> (matrixGPU, masksGPU, costsGPU, updatingCostsGPU, pathGPU, n, matrix_size);
		updateCosts <<<GRID_SIZE, BLOCK_SIZE >>> (costsGPU, updatingCostsGPU, masksGPU, matrix_size, doWorkGPU);
		cudaDeviceSynchronize();
	}

	//Copying result arrays to host
	cudaMemcpy(path, pathGPU, sizeof(int)*matrix_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(path, (costsGPU + matrix_size-1), sizeof(int), cudaMemcpyDeviceToHost);

	//Freeing memory
	free(masks);
	cudaFree(matrixGPU);
	cudaFree(costsGPU);
	cudaFree(updatingCostsGPU);
	cudaFree(pathGPU);
	cudaFree(masksGPU);
	cudaFreeHost(doWork);
	return path;
}

#undef BLOCK_SIZE
#undef GRID_SIZE
#undef INFINITY