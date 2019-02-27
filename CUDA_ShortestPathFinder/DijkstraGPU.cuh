#pragma once
__global__ void prepareArrays(int* costs, int* updatingCosts, bool* masks, int* path, int size);
__device__ void visitHorizontalNeighbor(int tIdx, int* matrix, int* cost, int* updatingCost, int* path, int n, int matrix_size);
__device__ void visitVerticalNeighbor(int tIdx, int* matrix, int* cost, int* updatingCost, int* path, int n, int matrix_size);
__global__ void visitNeighbors(int* matrix, bool* masks, int* cost, int* updatingCost, int* path, int n, int matrix_size);
__global__ void updateCosts(int* costs, int* updatingCosts, bool* masks, int matrix_size, bool *wasUpdated);
int* DijkstraGPU(int* matrix, int n, int m);