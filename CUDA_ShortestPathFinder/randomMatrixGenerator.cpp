#include "randomMatrixGenerator.h"
#include <random>
#include <iostream>
#include <chrono>

#define INFINITY 2<<27

//Generate matrix with random weights
int* generateRandomMatrix(int n, int m, int max_weight) {
	int* matrix = (int*)malloc(sizeof(int)*n*m);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine weight_generator(seed);
	for (int i = 0; i < n*m; i++) {
		matrix[i] = weight_generator() % (max_weight+1);
	}
	return matrix;
}

//Add walls impossible to cross at random places
void addWalls(int* matrix, int number_of_walls, int n, int m) {
	int max_wall_size = (n > m) ? n - 1 : m - 1;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rand(seed);
	while (number_of_walls--) {
		int size = rand() % max_wall_size + 1;
		int idx = rand() % (n*m - 1) + n;
		if (idx % n != 0 && idx % n != n - 1) {
			matrix[idx] = INFINITY;
			int direction = rand() % 2;
			while (--size) {
				switch (direction) {
				case 0:
					idx += 1;
					if (idx % n != 0 && idx % n != n - 1)
						matrix[idx] = INFINITY;
					break;
				case 1:
					idx += n;
					if (idx < n*m - n - 1)
						matrix[idx] = INFINITY;
				}
			}
		}
	}
}
#undef INFINITY
