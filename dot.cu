#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;
using namespace nvcuda;

__global__
void dot_TensorCore(float *a, float *b, float *c) {
	int32_t a_ldm = 16;
	int32_t b_ldm = 16;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;

	wmma::fill_fragment(c_frag, __float2half(0.f));
	
	__shared__ __half a_half[256] __align__(32);
	__shared__ __half b_half[256] __align__(32);
	__shared__ __half c_half[256] __align__(32);

	int32_t tid = threadIdx.x;

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
	//c_half[tid] = __float2half(1.f);
	__syncthreads();
	printf("share :: tid:%d, %f, %f, %f\n",tid, __half2float(a_half[tid]), __half2float(b_half[tid]), __half2float(c_half[tid]));

	a_frag.x[0] = __float2half(-1.f);
	printf("before load :: tid:%d, %f\n", tid, __half2float(a_frag.x[0]));
	wmma::load_matrix_sync(a_frag, a_half, a_ldm);
	printf("after load :: tid:%d, %f\n", tid, __half2float(a_frag.x[0]));

	wmma::load_matrix_sync(b_frag, b_half, b_ldm);

	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

	wmma::store_matrix_sync(c_half, c_frag, 16, wmma::mem_row_major);

	c[tid] = __half2float(c_half[tid]);
	printf("last :: tid:%d, %f\n", tid, c[tid]);
	
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
		a[i] = (float)i;
		b[i] = (float)i;
		c[i] = 0.f;
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
