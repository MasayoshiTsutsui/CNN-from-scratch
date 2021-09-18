#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

#define WARPSIZE 32
#define TILESIZE 16
#define ELEMS_TILE 256
#define TILEDIM_BLOCK 2 //1blockあたり、16*16の小行列タイルを2*2個生成する
#define TILES_BLOCK 4
using namespace std;
using namespace nvcuda;

//128スレッド4warpで起動されることを想定。2*2のタイルを1blockで計算
//タイルできれいに分割できない行列は未対応
__global__
void dot_TensorCore(float *a, float *b, float *c, int32_t m, int32_t n, int32_t k) {

	//a,b,cでは、小行列の要素が16個ごとにしか連続していない
	//shared memoryには、a,b,cから切り出した部分小行列の各要素が連続して並んでいる状況にする
	__shared__ __half a_half[ELEMS_TILE*TILES_BLOCK] __align__(32);
	__shared__ __half b_half[ELEMS_TILE*TILES_BLOCK] __align__(32);
	__shared__ __half c_half[ELEMS_TILE*TILES_BLOCK] __align__(32);


	int32_t lid = threadIdx.x % WARPSIZE; //warp内の識別id
	int32_t lid_hex = lid % 16;
	int32_t hexid = lid / 16;
	int32_t wid = threadIdx.x / WARPSIZE;
	int32_t tileIdx_x = blockIdx.x * TILEDIM_BLOCK + wid % 2; // 自スレッドがcのx軸方向何枚目のタイル生成担当か
	int32_t tileIdx_y = blockIdx.y * TILEDIM_BLOCK + wid / 2; // 自スレッドがcのy軸以下略

	wmma::fragment<wmma::matrix_a, TILESIZE, TILESIZE, TILESIZE, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, TILESIZE, TILESIZE, TILESIZE, __half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, TILESIZE, TILESIZE, TILESIZE, __half> c_frag;

	wmma::fill_fragment(c_frag, __float2half(0.f));
	//16*16*16で分割した時にはみ出た部分も余分に計算するために、iの終点を工夫

	if ((tileIdx_y * TILESIZE <= m) && (tileIdx_x * TILESIZE <= n)) { //そもそもcの完全に外に行っているタイルは計算しない

		if(((tileIdx_y + 1) * TILESIZE > m) && ((tileIdx_x + 1) * TILESIZE > n)) { //担当するcのタイルが下にも右にもcからはみ出しているとき

			for (int32_t i=0; i < (k-1) / TILESIZE + 1; i++) {
				if ((i+1) * TILESIZE <= k) { //処理しようとしているタイルがa,bの中に(k方向には)収まっているとき

					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (tileIdx_y * TILESIZE + 2*j + hexid < m) { //自スレッドが処理中の行が、まだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (tileIdx_x * TILESIZE + lid_hex < n) { //自スレッドが処理中の列が、まだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				else { //ループの最後でa,bから(k方向に)はみ出してしまった時
					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if ((tileIdx_y * TILESIZE + 2*j + hexid < m) && (i * TILESIZE + lid_hex < k)) { //自スレッドが処理中の要素が、行方向にも列方向にもまだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if ((tileIdx_x * TILESIZE + lid_hex < n) && (i * TILESIZE + 2*j + hexid < k)) { //自スレッドが処理中の要素が、行方向にも列方向にもまだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				wmma::load_matrix_sync(a_frag, &a_half[wid*ELEMS_TILE], 16);
				wmma::load_matrix_sync(b_frag, &b_half[wid*ELEMS_TILE], 16);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			wmma::store_matrix_sync(&c_half[wid*ELEMS_TILE], c_frag, 16, wmma::mem_row_major);

			int32_t c_offsetbase = tileIdx_y * TILESIZE * n + tileIdx_x * TILESIZE;
			for (int32_t j=0; j < TILESIZE / 2; j++) {
				if ((tileIdx_y * TILESIZE + 2*j + hexid < m) && (tileIdx_x * TILESIZE + lid_hex < n)) { //自スレッドが処理中の要素が、行方向にも列方向にもまだcの中に収まってる場合だけstore
					c[c_offsetbase + hexid*n+lid_hex] = __half2float(c_half[wid*ELEMS_TILE + lid + j*32]);
				}
				c_offsetbase += 2 * n; //2行下に移動
			}
		}
		else if ((tileIdx_y + 1) * TILESIZE > m) { //担当するcのタイルが下にだけはみ出している
			for (int32_t i=0; i < (k-1) / TILESIZE + 1; i++) {
				if ((i+1) * TILESIZE <= k) { //処理しようとしているタイルがa,bの中に(k方向には)収まっているとき

					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (tileIdx_y * TILESIZE + 2*j + hexid < m) { //自スレッドが処理中の行が、まだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				else { //ループの最後でa,bから(k方向に)はみ出してしまった時
					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if ((tileIdx_y * TILESIZE + 2*j + hexid < m) && (i * TILESIZE + lid_hex < k)) { //自スレッドが処理中の要素が、行方向にも列方向にもまだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (i * TILESIZE + 2*j + hexid < k) { //自スレッドが処理中の要素が列方向にまだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				wmma::load_matrix_sync(a_frag, &a_half[wid*ELEMS_TILE], 16);
				wmma::load_matrix_sync(b_frag, &b_half[wid*ELEMS_TILE], 16);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			wmma::store_matrix_sync(&c_half[wid*ELEMS_TILE], c_frag, 16, wmma::mem_row_major);

			int32_t c_offsetbase = tileIdx_y * TILESIZE * n + tileIdx_x * TILESIZE;
			for (int32_t j=0; j < TILESIZE / 2; j++) {
				if (tileIdx_y * TILESIZE + 2*j + hexid < m) { //自スレッドが処理中の要素が、列方向にまだcの中に収まってる場合だけstore
					c[c_offsetbase + hexid*n+lid_hex] = __half2float(c_half[wid*ELEMS_TILE + lid + j*32]);
				}
				c_offsetbase += 2 * n; //2行下に移動
			}
		}
		else if ((tileIdx_x + 1) * TILESIZE > n) { //右にだけはみ出している
			for (int32_t i=0; i < (k-1) / TILESIZE + 1; i++) {
				if ((i+1) * TILESIZE <= k) { //処理しようとしているタイルがa,bの中に(k方向には)収まっているとき

					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (tileIdx_x * TILESIZE + lid_hex < n) { //自スレッドが処理中の列が、まだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				else { //ループの最後でa,bから(k方向に)はみ出してしまった時
					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE; //a,bの中でのタイルの先頭要素のidx
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (i * TILESIZE + lid_hex < k) { //自スレッドが処理中の要素が、行方向にまだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if ((tileIdx_x * TILESIZE + lid_hex < n) && (i * TILESIZE + 2*j + hexid < k)) { //自スレッドが処理中の要素が、行方向にも列方向にもまだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				wmma::load_matrix_sync(a_frag, &a_half[wid*ELEMS_TILE], 16);
				wmma::load_matrix_sync(b_frag, &b_half[wid*ELEMS_TILE], 16);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			wmma::store_matrix_sync(&c_half[wid*ELEMS_TILE], c_frag, 16, wmma::mem_row_major);

			int32_t c_offsetbase = tileIdx_y * TILESIZE * n + tileIdx_x * TILESIZE;
			for (int32_t j=0; j < TILESIZE / 2; j++) {
				if (tileIdx_x * TILESIZE + lid_hex < n) { //自スレッドが処理中の要素が、行方向にも列方向にもまだcの中に収まってる場合だけstore
					c[c_offsetbase + hexid*n+lid_hex] = __half2float(c_half[wid*ELEMS_TILE + lid + j*32]);
				}
				c_offsetbase += 2 * n; //2行下に移動
			}
		}
		else { //はみ出してない
			for (int32_t i=0; i < (k-1) / TILESIZE + 1; i++) {
				if ((i+1) * TILESIZE <= k) { //処理しようとしているタイルがa,bの中に収まっているとき
					//a,bの中でのタイルの先頭要素のidx
					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE;
					//16*16*16でやろうとしてるので、tidが0~15の担当要素、16~31の担当要素は隔たりがある
					//1回で小行列の2行分をa_halfに。
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				else { //このループで最後に16*16のタイル分割時にa,bからはみ出してしまった時
					int32_t a_offsetbase = tileIdx_y * TILESIZE * k + i * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (i * TILESIZE + lid_hex < k) { //自スレッドが処理中の要素が、行方向にまだaの中に収まってる場合は、aからデータをload
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(a[a_offsetbase + hexid*k+lid_hex]);
						}
						else { //aに収まってない場合は0埋め
							a_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						a_offsetbase += 2 * k; //2行下に移動
					}

					int32_t b_offsetbase = i * TILESIZE * n + tileIdx_x * TILESIZE;
					for (int32_t j=0; j < TILESIZE / 2; j++) {
						if (i * TILESIZE + 2*j + hexid < k) { //自スレッドが処理中の要素が、行方向にも列方向にもまだbの中に収まってる場合は、bからデータをload
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(b[b_offsetbase + hexid*n+lid_hex]);
						}
						else { //bに収まってない場合は0埋め
							b_half[wid*ELEMS_TILE + lid + j*32] = __float2half(0.);
						}
						b_offsetbase += 2 * n; //2行下に移動
					}
				}
				wmma::load_matrix_sync(a_frag, &a_half[wid*ELEMS_TILE], 16);
				wmma::load_matrix_sync(b_frag, &b_half[wid*ELEMS_TILE], 16);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			wmma::store_matrix_sync(&c_half[wid*ELEMS_TILE], c_frag, 16, wmma::mem_row_major);

			int32_t c_offsetbase = tileIdx_y * TILESIZE * n + tileIdx_x * TILESIZE;
			for (int32_t j=0; j < TILESIZE / 2; j++) {
				c[c_offsetbase + hexid*n+lid_hex] = __half2float(c_half[wid*ELEMS_TILE + lid + j*32]);
				c_offsetbase += 2 * n; //2行下に移動
			}
		}
	}
}

int main() {
	int32_t m = 33;
	int32_t k = 36;
	int32_t n = 34;
	int32_t a_matsize = m * k;
	int32_t b_matsize = k * n;
	int32_t c_matsize = m * n;
	float *a, *b, *c;
	float *a_dev, *b_dev, *c_dev;
	a = (float*)malloc(sizeof(float) * a_matsize);
	b = (float*)malloc(sizeof(float) * b_matsize);
	c = (float*)malloc(sizeof(float) * c_matsize);

	cudaMalloc((void**)&a_dev, sizeof(float) * a_matsize);
	cudaMalloc((void**)&b_dev, sizeof(float) * b_matsize);
	cudaMalloc((void**)&c_dev, sizeof(float) * c_matsize);
	for (int32_t i=0; i < a_matsize; i++) {
		a[i] = (float)i;
	}
	for (int32_t i=0; i < b_matsize; i++) {
		b[i] = 0.;
	}
	for (int32_t i=0; i < c_matsize; i++) {
		c[i] = 0.;
	}
	for (int32_t i=0; i < n; i++) {
		b[i] = (float)i;
	}
	cudaMemcpy(a_dev, a, sizeof(float)*a_matsize, cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, sizeof(float)*b_matsize, cudaMemcpyHostToDevice);
	dim3 grid;
	grid.x = 2;
	grid.y = 2;
	dot_TensorCore<<<grid, 128>>>(a_dev,  b_dev, c_dev, m, n, k);
	cudaMemcpy(c, c_dev, sizeof(float)*c_matsize, cudaMemcpyDeviceToHost);
	for (int32_t i=0; i < m; i++) {
		for (int32_t j=0; j < n; j++) {
			cout << c[i*n+j] << " ";
		}
		cout << endl;
	}
	return 0;
}
