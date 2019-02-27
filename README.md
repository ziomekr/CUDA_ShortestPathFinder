# CUDA_ShortestPathFinder
Project written for CUDA programming WUT CS course
## Problem description

Given N x M matrix, where each field has positive integer weight value, find a shortest path from top left corner to bottom right corner.
From any field one can move up, down, right and left (with restriction to border fields, i.e. one cannot move left from left border field
etc.) On the matrix there are placed walls, which cannot be crossed (in practice, they have infinite weight). Problem is solved using Dijkstra
algorithm.

## CPU implementation
For the best performance for the CPU I've implemented Fibonacci Heap to serve as priority queue in the algorithm. 
## GPU implementation

Parallelization of the Dijkstra algorithm is efficient when the graph is dense, so here, where each field has at most 4 neighbors is not so effective
and turned out to be slower than my CPU implementation.
Parallelization occurs in such a way, that all fields with cost to reach updated in previous run are visited simultaneously. 

## Usage
Tests are included in "ShortestPathFinder.cu" file. Test function usage is following: test(n, m, number, max_weight)
- n, m - matrix dimensions
- number - arbitrary test number
- max_weight - maximum weight for each field (all the weights are generated randomly, with max = max_weight)
Test function also produce bitmap of raw matrix without marked path, one with path found by CPU and by GPU. 
