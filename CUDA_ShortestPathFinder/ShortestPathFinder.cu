//Testing, resulting in bitmap with marked path, CPU and GPU exec time and costs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DijkstraCPU.h"
#include "DijkstraGPU.cuh"
#include "randomMatrixGenerator.h"
#include "visualization.cuh"
#include <algorithm>
#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <string>
#define MAX_WALLS 30

void test(int n, int m, int testNumber, int maxWeight) {
	std::cout << "TEST " << testNumber << std::endl;
	int* matrix  = generateRandomMatrix(n, m, maxWeight);
	int* matrix1 = (int*)malloc(sizeof(int)*n*m);
	const std::string raw = "raw" + std::to_string(testNumber) + ".bmp";
	const std::string cpu = "cpu" + std::to_string(testNumber) + ".bmp";
	const std::string gpu = "gpu" + std::to_string(testNumber) + ".bmp";
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rand(seed);
	int walls = (rand() % (n/3)) + 1;
	walls = (walls > MAX_WALLS) ? MAX_WALLS : walls;
	addWalls(matrix, walls, n, m);
	std::copy(matrix, matrix + n * m - 1, matrix1);
	std::clock_t c_start = std::clock();
	std::vector<int> p = Dijkstra(matrix, n, m);
	int* pathCPU = &p[0];
	std::clock_t c_end = std::clock();
	long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	std::cout << "Path length CPU: " << pathCPU[0] << std::endl;
	std::cout << "Time: " << time_elapsed_ms << " ms\n";
	c_start = std::clock();
	int* pathGPU = DijkstraGPU(matrix, n, m);
	c_end = std::clock();
	time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	std::cout << "Path length GPU: " << pathGPU[0] << std::endl;
	std::cout << "Time: " << time_elapsed_ms << " ms\n";
	visualizeMatrix(raw, matrix, n, m, 16, 0, maxWeight);
	addPathToMatrix(matrix, pathCPU, n*m);
	addPathToMatrix(matrix1, pathGPU, n*m);
	visualizeMatrix(cpu, matrix, n, m, 16, 0, maxWeight);
	visualizeMatrix(gpu, matrix1, n, m, 16, 0, maxWeight);
	free(matrix);
	free(matrix1);
	free(pathGPU);
}

int main() {
	test(10, 10, 1, 15);
	test(100, 100, 2, 25);
	test(1000, 1000, 3, 10);
	return 0;
}
