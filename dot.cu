#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;
using namespace nvcuda;

__global__
void dot_TensorCore(float *a, float *b, float *c) {

	__shared__ __half a_half[257] __align__(32);
	__shared__ __half b_half[258] __align__(32);
	__shared__ __half c_half[259] __align__(32);

	int32_t tid = threadIdx.x;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;

	wmma::fill_fragment(c_frag, __float2half(0.f));

	a_half[tid] = __float2half(a[tid]);
	a_half[tid+32] = __float2half(a[tid+32]);
	a_half[tid+64] = __float2half(a[tid+64]);
	a_half[tid+96] = __float2half(a[tid+96]);
	a_half[tid+128] = __float2half(a[tid+128]);
	a_half[tid+160] = __float2half(a[tid+160]);
	a_half[tid+192] = __float2half(a[tid+192]);
	a_half[tid+224] = __float2half(a[tid+224]);
	b_half[tid] = __float2half(b[tid]);
	b_half[tid+32] = __float2half(b[tid+32]);
	b_half[tid+64] = __float2half(b[tid+64]);
	b_half[tid+96] = __float2half(b[tid+96]);
	b_half[tid+128] = __float2half(b[tid+128]);
	b_half[tid+160] = __float2half(b[tid+160]);
	b_half[tid+192] = __float2half(b[tid+192]);
	b_half[tid+224] = __float2half(b[tid+224]);

	wmma::load_matrix_sync(a_frag, a_half, 16);
	wmma::load_matrix_sync(b_frag, b_half, 16);

	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

	wmma::store_matrix_sync(c_half, c_frag, 16, wmma::mem_row_major);

	c[tid] = __half2float(c_half[tid]);
	c[tid+32] = __half2float(c_half[tid+32]);
	c[tid+64] = __half2float(c_half[tid+64]);
	c[tid+96] = __half2float(c_half[tid+96]);
	c[tid+128] = __half2float(c_half[tid+128]);
	c[tid+160] = __half2float(c_half[tid+160]);
	c[tid+192] = __half2float(c_half[tid+192]);
	c[tid+224] = __half2float(c_half[tid+224]);
}

int main() {
	int32_t n = 16;
	int32_t matsize = n * n;
	float *a, *b, *c;
	float *a_dev, *b_dev, *c_dev;
	a = (float*)malloc(sizeof(float) * matsize);
	b = (float*)malloc(sizeof(float) * matsize);
	c = (float*)malloc(sizeof(float) * matsize);

	cudaMalloc((void**)&a_dev, sizeof(float) * matsize);
	cudaMalloc((void**)&b_dev, sizeof(float) * matsize);
	cudaMalloc((void**)&c_dev, sizeof(float) * matsize);
	for (int32_t i=0; i < matsize; i++) {
		a[i] = 1.;
		b[i] = 0.;
		c[i] = 0.;
	}
	for (int32_t i=0; i < n; i++) {
		b[i] = 1.;
	}
	cudaMemcpy(a_dev, a, sizeof(float)*matsize, cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, sizeof(float)*matsize, cudaMemcpyHostToDevice);
	dot_TensorCore<<<1, 32>>>(a_dev,  b_dev, c_dev);
	cudaMemcpy(c, c_dev, sizeof(float)*matsize, cudaMemcpyDeviceToHost);
	for (int32_t i=0; i < n; i++) {
		for (int32_t j=0; j < n; j++) {
			cout << c[i*n+j] << " ";
		}
		cout << endl;
	}
	return 0;
}
