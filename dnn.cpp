#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "tensor.hpp"
#include "dnn.hpp"

using namespace std;

//extern void dotTC_invoker(Tensor &a, Tensor &b, Tensor &c, int32_t m, int32_t n, int32_t k, int32_t TorN);

void affine_layer(Tensor &x, Tensor &weight, Tensor &bias, Tensor &z) {
	dot(x, weight, z, NandN);
	add_bias(z, bias);
}

void sigmoid(Tensor &a) {
	#pragma acc kernels present(a)
	#pragma acc loop independent gang
	for (int32_t i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			a[i*a.w + j] = 1. / (1. + exp(-a[i*a.w + j]));
		}
	}
}

//バイト列からintへの変換
uint32_t reverseInt (uint32_t i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

void readTrainingFile(string filename, Tensor &images){
	ifstream ifs(filename.c_str(),std::ios::in | std::ios::binary);
	if (!ifs)
    {
        cout << "ファイルが開けませんでした。" << std::endl;
        return;
    }
	int32_t magic_number = 0;
	int32_t number_of_images = 0;
	int32_t rows = 0;
	int32_t cols = 0;


	//ヘッダー部より情報を読取る。
	ifs.read((char*)&magic_number,sizeof(magic_number)); 
	magic_number= reverseInt(magic_number);
	ifs.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= reverseInt(number_of_images);
	ifs.read((char*)&rows,sizeof(rows));
	rows= reverseInt(rows);
	ifs.read((char*)&cols,sizeof(cols));
	cols= reverseInt(cols);

	images.SetDim(1, 1, number_of_images, rows*cols);


	for(int32_t i = 0; i < number_of_images; i++){

		for(int32_t row = 0; row < rows; row++){
			for(int32_t col = 0; col < cols; col++){
				unsigned char temp = 0;
				ifs.read((char*)&temp,sizeof(temp));
				images[i*rows*cols + row*cols + col] = (float)temp;
			}
		}
	}
}

//ラベルを読み込み、one-hotに変換
void readLabelFile(string filename, Tensor &label){
	ifstream ifs(filename.c_str(),std::ios::in | std::ios::binary);
	if (!ifs)
    {
        cout << "ファイルが開けませんでした。" << std::endl;
        return;
    }
	int32_t magic_number = 0;
	int32_t number_of_images = 0;

	//ヘッダー部より情報を読取る。
	ifs.read((char*)&magic_number,sizeof(magic_number)); 
	magic_number= reverseInt(magic_number);
	ifs.read((char*)&number_of_images,sizeof(number_of_images));
	number_of_images= reverseInt(number_of_images);

	label.SetDim(1, 1, number_of_images, 10);


	init_zero(label);

	for(int32_t i = 0; i < number_of_images; i++){
		unsigned char temp = 0;
		ifs.read((char*)&temp, sizeof(temp));
		if ((int32_t)temp > 9 || (int32_t)temp < 0) {
			cout << "label is not 0-9 digits!" << endl;
			return;
		}
		label[i*10 + (int32_t)temp] = 1.;
	}
}

//void dotTC(Tensor &a, Tensor &b, Tensor &c, int32_t TorN) {
	//int32_t m, k, k_, n;
	//switch(TorN) {
		//case NandN:
			//m = a.h; k = a.w; k_ = b.h; n = b.w;
			//break;
		//case TandN:
			//m = a.w; k = a.h; k_ = b.h; n = b.w;
			//break;
		//case NandT:
			//m = a.h; k = a.w; k_ = b.w; n = b.h;
			//break;
		//case TandT:
			//m = a.w; k = a.h; k_ = b.w; n = b.h;
			//break;
		//default:
			//cout << "Transpose error in dotTC." << endl;
			//return;
	//}

	//if (k != k_ || m != c.h || n != c.w) {
		//cout << "tensor size mismatch in dotTC_invoker." << endl << endl;
		//return;
	//}

	//#pragma acc host_data use_device(a, b, c)
	//{
		//dotTC_invoker(a, b, c, m, n, k, TorN);
	//}

//}
// m*k & k_*n matrix multiplication
//TorNは転置指定子
void dot(Tensor &a, Tensor &b, Tensor &c, int32_t TorN) {
	int32_t m;
	int32_t k;
	int32_t k_;
	int32_t n;
	switch(TorN) {
		case NandN:
			m = a.h;
			k = a.w;
			k_ = b.h;
			n = b.w;
			break;
		case TandN:
			m = a.w;
			k = a.h;
			k_ = b.h;
			n = b.w;
			break;
		case NandT:
			m = a.h;
			k = a.w;
			k_ = b.w;
			n = b.h;
			break;
		case TandT:
			m = a.w;
			k = a.h;
			k_ = b.w;
			n = b.h;
			break;
		default:
			cout << "Transpose error in dot." << endl;
			return;
	}
	if (k != k_ || m != c.h || n != c.w) {
		cout << "tensor size mismatch in dot." << endl << endl;
		return;
	}

	switch(TorN) {
		case NandN:
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (int32_t i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (int32_t j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (int32_t x=0; x < k; x++) {
						c[i*n+j] += a[i*k+x] * b[x*n+j];
					}
				}
			}
			break;
		case TandN:
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (int32_t i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (int32_t j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (int32_t x=0; x < k; x++) {
						c[i*n+j] += a[m*x+i] * b[x*n+j];
					}
				}
			}
			break;
		case NandT:
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (int32_t i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (int32_t j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (int32_t x=0; x < k; x++) {
						c[i*n+j] += a[i*k+x] * b[k*j+x];
					}
				}
			}
			break;
		case TandT:
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (int32_t i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (int32_t j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (int32_t x=0; x < k; x++) {
						c[i*n+j] += a[m*x+i] * b[k*j+x];
					}
				}
			}
			break;
	}
}

//行列aの各行にベクトルbを足しこむ
void add_bias(Tensor &a, Tensor &b) {
	if (a.w != b.w) {
		cout << "Tensor size mismatch in add." << endl;
		return;
	}
	#pragma acc kernels present(a, b)
	#pragma acc loop independent gang
	for (int32_t i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			a[i*a.w + j] += b[j];
		}
	}
}

//a-scale*bをcに代入
void scale_sub(Tensor &a, Tensor &b, Tensor &c, float scale) {
	if (a.h != b.h || a.w != b.w || a.h != c.h || a.w != c.w) {
		cout << "Tensor size mismatch in sub." << endl;
		return;
	}
	#pragma acc kernels present(a, b, c)
	#pragma acc loop independent gang
	for (int32_t i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			c[i*a.w + j] = a[i*a.w + j] - scale*b[i*a.w + j];
		}
	}
}

void div_by_scalar(Tensor &a, float d) {
	#pragma acc kernels present(a)
	#pragma acc loop independent gang
	for (int32_t i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			a[i*a.w + j] = a[i*a.w + j] / d;
		}
	}
}

void relu(Tensor &s, Tensor &t) {
	if (s.h != 1 || t.h != 1) {
		cout << "relu only receives vector." << endl << endl;
		return;
	}
	int32_t l = s.w;
	for (int32_t i=0; i < l; i++) {
		t[i] = (s[i] < 0.) ? 0. : s[i];
	}
}

void init_random(Tensor &w) {
	float sigma = 0.01;
	float mean = 0.0;
	int32_t height = w.h;
	int32_t width = w.w;
	random_device seed_gen;
	default_random_engine engine(seed_gen());

	normal_distribution<> dist(mean, sigma);

	for (int32_t i=0; i < height; i++) {
		for (int32_t j=0; j < width; j++) {
			w[i*width + j] = dist(engine);
		}
	}
}

void init_zero(Tensor &w) {
	int32_t height = w.h;
	int32_t width = w.w;

	for (int32_t i=0; i < height; i++) {
		for (int32_t j=0; j < width; j++) {
			w[i*width + j] = 0.;
		}
	}
}


void softmax(Tensor &a) {
	float max_pxl;
	float sum_exp;
	#pragma acc paraint32_tel present(a) private(max_pxl, sum_exp)
	#pragma acc loop independent gang
	for (int32_t i=0; i < a.h; i++) {
		max_pxl = -1000000.;
		#pragma acc loop vector reduction( max : max_pxl )
		for (int32_t j=0; j < a.w; j++) {
			max_pxl = max(max_pxl, a[i*a.w + j]);
		}
		sum_exp = 0.;

		float exp_a_c;
		#pragma acc loop vector reduction( + : sum_exp)
		for (int32_t j=0; j < a.w; j++) {
			exp_a_c = exp(a[i*a.w + j] - max_pxl);
			a[i*a.w + j] = exp_a_c;
			sum_exp += exp_a_c;
		}
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			a[i*a.w + j] /= sum_exp;
		}
	}
}

//クロスエントロピー誤差関数
double loss(Tensor &y, Tensor &t) {
	if (y.h != t.h || y.w != t.w) {
		cout << "Tensor size mismatch in loss." << endl;
		return -1.;
	}

	double delta = 0.0000001;
	double los = 0.;

	for (int32_t i=0; i < y.h; i++) {
		for (int32_t j=0; j < y.w; j++) {
			los += -t[i*y.w + j] * log(y[i*y.w + j] + delta);
		}
	}
	los /= y.h;
	return los;
}

//行列を縦方向に和を取り、ベクトルにする
void sum_vertical(Tensor &a, Tensor &v) {
	if (v.h != 1 || a.w != v.w) {
		cout << "Tensor size mismatch in sum_vertical." << endl;
		return;
	}
	init_zero(v);
	#pragma acc kernels present(a, v)
	#pragma acc loop seq
	for (int32_t i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < a.w; j++) {
			v[j] += a[i*a.w + j];
		}
	}
}



//dz行列の各要素に、z、(1-z)の対応する各要素をかけていく
void back_sigmoid(Tensor &dz, Tensor &z) {
	if (dz.h != z.h || dz.w != z.w) {
		cout << "Tensor size mismatch in back_sigmoid." << endl;
		return;
	}

	#pragma acc kernels present(dz, z)
	#pragma acc loop independent gang
	for (int32_t i=0; i < dz.h; i++) {
		#pragma acc loop independent vector
		for (int32_t j=0; j < dz.w; j++) {
			dz[i*dz.w+j] *= z[i*dz.w+j] * (1 - z[i*dz.w+j]);
		}
	}
}

float accuracy(Tensor &y, Tensor &t) {
	if(y.h != t.h || y.w != t.w) {
		cout << "Tensor size mismatch in accuracy." << endl;
		return -1.;
	}
	int32_t ymax_idx;
	int32_t tmax_idx;
	float ymax;
	float tmax;
	float acc = 0.;

	#pragma acc paraint32_tel present(y, t)
	#pragma acc loop independent vector reduction( + : acc)
	for (int32_t i=0; i < y.h; i++) {
		ymax_idx = -1;
		tmax_idx = -1;
		ymax = -1.;
		tmax = -1.;
		#pragma acc loop seq
		for (int32_t j=0; j < y.w; j++) {
			if (ymax < y[i*y.w+j]) {
				ymax_idx = j;
				ymax = y[i*y.w+j];
			}
			if (tmax < t[i*y.w+j]) {
				tmax_idx = j;
				tmax = t[i*y.w+j];
			}
		}
		if (ymax_idx == tmax_idx) {
			acc += 1.;
		}
	}
	return acc / y.h;
}

void batch_random_choice(Tensor &dataset, Tensor &labelset, Tensor &x, Tensor &t) {
	random_device rnd;
	mt19937 mt(rnd());
	uniform_int_distribution<> randbatch(0, dataset.h-1);
	init_zero(t);
	for (int32_t i=0; i < x.h; i++) {
		int32_t img_idx = randbatch(mt);

		for (int32_t j=0; j < 10; j++) {
			t[i*10 + j] = labelset[img_idx*10 + j];
		}
		for (int32_t j=0; j < x.w; j++) {
			x[i*x.w + j] = dataset[img_idx*x.w + j];
		}
	}
}