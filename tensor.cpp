#include <iostream>
#include <random>
#include "tensor.hpp"


void init_random(Tensor &a) {
	float sigma = 0.01;
	float mean = 0.0;
	int32_t datanum = a.d;
	int32_t channel = a.c;
	int32_t height = a.h;
	int32_t width = a.w;
	random_device seed_gen;
	default_random_engine engine(seed_gen());

	normal_distribution<> dist(mean, sigma);

	int32_t idx = 0;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				for (int32_t w=0; w < width; w++) {
					a[idx] = dist(engine);
					idx++;
				}
			}
		}
	}
}

void init_zero(Tensor &a) {
	int32_t datanum = a.d;
	int32_t channel = a.c;
	int32_t height = a.h;
	int32_t width = a.w;

	int32_t idx = 0;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				for (int32_t w=0; w < width; w++) {
					a[idx] = 0.;
					idx++;
				}
			}
		}
	}
}

void init_zeroint(IntTensor &a) {
	int32_t datanum = a.d;
	int32_t channel = a.c;
	int32_t height = a.h;
	int32_t width = a.w;

	int32_t idx = 0;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				for (int32_t w=0; w < width; w++) {
					a[idx] = 0;
					idx++;
				}
			}
		}
	}
}

