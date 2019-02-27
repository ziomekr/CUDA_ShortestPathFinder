#pragma once
#include "FibonacciHeap.h"
#include <vector>
FibonacciHeap makePriorityQueue(std::vector<FibonacciHeap::Node*> nodes);
std::vector<FibonacciHeap::Node*> makeQueueNodes(int n, int m);
std::vector<int> Dijkstra(int* matrix, int n, int m);