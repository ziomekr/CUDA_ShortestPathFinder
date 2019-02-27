//CPU version of Dijkstra algorithm

#include "DijkstraCPU.h"
#include "FibonacciHeap.h"
#include <stdlib.h>
#include <vector>

#define INFINITY 2<<27

//Creating heap nodes for nxm matrix
std::vector<FibonacciHeap::Node*> makeQueueNodes(int n, int m) {
	std::vector<FibonacciHeap::Node*> vertices(n*m, nullptr);
	vertices[0] = new FibonacciHeap::Node(0, 0);
	for (int i = 1; i < n*m; i++) {
		FibonacciHeap::Node* node = new FibonacciHeap::Node(i, INFINITY);
		vertices[i] = node;
	}
	return vertices;
}

//Create priority queue from nodes
FibonacciHeap makePriorityQueue(std::vector<FibonacciHeap::Node*> nodes) {
	FibonacciHeap queue;
	for each (FibonacciHeap::Node* n in nodes)
	{
		queue.Insert(n);
	}
	return queue;
}

std::vector<int> Dijkstra(int* matrix, int n, int m) {
	//Initialization
	std::vector<bool> visited(n*m, false);
	std::vector<FibonacciHeap::Node*> vertices = makeQueueNodes(n,m);
	std::vector<int> path(n*m, -1);
	FibonacciHeap priorityQueue = makePriorityQueue(vertices);
	FibonacciHeap::Node* current;

	//Actual Dijskstra, proceed until all nodes were visited
	while (priorityQueue.size!=0) {
		current = priorityQueue.extractMinimum();
		FibonacciHeap::Node* adjacent;
		int new_weight;
		int idx = current->key;
		for (int i = -1; i < 2; i += 2) {
			if (((idx + i) / n == idx / n) && (idx + i > -1) && (idx+i <n*m)){
				if (!visited[idx + i]) {
					adjacent = vertices[idx + i];
					new_weight = current->value + matrix[idx + i];
					if (adjacent->value > new_weight) {
						priorityQueue.decreaseValue(adjacent, new_weight);
						path[idx + i] = idx;
					}
				}
			}
			if (((idx + n * i)>-1 ) && ((idx + n * i)<n*m)) {
				if (!visited[idx + n*i]) {
					adjacent = vertices[idx + n*i];
					new_weight = current->value + matrix[idx + n*i];
					if (adjacent->value > new_weight) {
						priorityQueue.decreaseValue(adjacent, new_weight);
						path[idx + n*i] = idx;
					}
				}
			}
		}
		visited[idx] = true;
	}
	path[0] = vertices[n*m-1]->value;
	return path;
}
#undef INFINITY
