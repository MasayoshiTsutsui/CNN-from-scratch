#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;
using ll = int64_t;

int32_t batch_size = 100;
int32_t feature_size1 = 100;
int32_t iters_num = 1201;
double learning_rate = 0.1;


int32_t image_size;
int32_t label_size;

#define NandN 0
#define TandN 1
#define NandT 2
#define TandT 3


class Tensor
{
	private:
		double* _A{nullptr};

	public:

		int32_t h{0};
		int32_t w{0};
		int32_t size{0};

		#pragma acc routine seq
		double& operator[](size_t idx) { return _A[idx]; };
		
		explicit Tensor() { };
		//コンストラクタでもうgpu側にメモリ領域を確保してしまう
		explicit Tensor(int32_t height, int32_t width) {
			h = height; w = width; size = height * width;
			_A = new double[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(_A[0:size])
		}
		~Tensor() {
			#pragma acc exit data delete(_A[0:size])
			#pragma acc exit data delete(this)
			delete [] _A;
			_A = NULL;
			h = 0; w = 0; size = 0;
		}

		inline void updateHost() {
			#pragma acc update self(_A[0:size])
		}
		inline void updateDev() {
			#pragma acc update device(_A[0:size])
		}

		void Print() {
			for (ll i=0; i < h; i++) {
				for (ll j=0; j < w; j++) {
					cout << _A[i*w+j] << " ";
				}
				cout << endl;
			}
		}
		void SetDim(int32_t height, int32_t width) {
			h = height; w = width; size = height * width;
			_A = new double[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(_A[0:size])
		}
		//void FreeVal();
};

//Tensor::Tensor() {
	//h = 0; w = 0;
//}
//Tensor::Tensor(int32_t height, int32_t width) {
	//h = height; w = width;
	//val = (double*)malloc(sizeof(double) * height * width);
//}

//void Tensor::Print() {
	//for (ll i=0; i < h; i++) {
		//for (ll j=0; j < w; j++) {
			//cout << val[i*w+j] << " ";
		//}
		//cout << endl;
	//}
//}

//void Tensor::SetDim(int32_t height, int32_t width) {
	//h = height; w = width;
	//val = (double*)malloc(sizeof(double) * height * width);
//}

//void Tensor::FreeVal() {
	//free(val);
//}

uint32_t reverseInt (uint32_t i);
void sigmoid(Tensor &a);
void relu(Tensor &s, Tensor &t);
void dot(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
void add_bias(Tensor &a, Tensor &b); //aは行列、bはベクトル。aの各行にbを足しこむ
void scale_sub(Tensor &a, Tensor &b, Tensor &c, double scale); //a-scale*b
uint32_t reverseInt (uint32_t i);
void readTrainingFile(string filename, Tensor &images);
void readLabelFile(string filename, Tensor &label);
void init_random(Tensor &w);
void init_zero(Tensor &w);
void batch_random_choice(Tensor &dataset, Tensor &labelset, Tensor &x, Tensor &t);
void softmax(Tensor &a);
double loss(Tensor &y, Tensor &t);
void div_by_scalar(Tensor &a, double d);
void sum_vertical(Tensor &a, Tensor &v);
void back_sigmoid(Tensor &dz, Tensor &z);
double accuracy(Tensor &y, Tensor &t);

void affine_layer(Tensor &x, Tensor &weight, Tensor &bias, Tensor &z) {
	dot(x, weight, z, NandN);
	add_bias(z, bias);
}

int main() {

	Tensor train_data, train_label;
	Tensor test_data, test_label;

	readTrainingFile("./mnist/train-images-idx3-ubyte", train_data);
	readLabelFile("./mnist/train-labels-idx1-ubyte", train_label); //one-hot label
	readTrainingFile("./mnist/t10k-images-idx3-ubyte", test_data);
	readLabelFile("./mnist/t10k-labels-idx1-ubyte", test_label); //one-hot label
	
	test_data.updateDev();
	test_label.updateDev();


	int32_t train_size = train_data.h;
	int32_t test_size = test_data.h;
	int32_t iter_per_epoch = train_size / batch_size;

	image_size = train_data.w;
	label_size = train_label.w;

	Tensor x(batch_size, image_size);
	Tensor t(batch_size, label_size);

	Tensor w1(image_size, feature_size1);
	Tensor dw1(image_size, feature_size1);
	Tensor b1(1, feature_size1);
	Tensor db1(1, feature_size1);
	

	Tensor z1(batch_size, feature_size1);
	Tensor dz1(batch_size, feature_size1);

	Tensor z1_test(test_size, feature_size1);

	Tensor w2(feature_size1, label_size);
	Tensor dw2(feature_size1, label_size);
	Tensor b2(1, label_size);
	Tensor db2(1, label_size);

	Tensor y(batch_size, label_size);
	Tensor dy(batch_size, label_size);

	Tensor y_test(test_size, label_size);

	init_zero(x);
	init_zero(t);

	init_random(w1);
	w1.updateDev();
	init_zero(dw1);

	init_zero(b1);
	b1.updateDev();
	init_zero(db1);

	init_zero(z1);
	init_zero(dz1);

	init_zero(z1_test);

	init_random(w2);
	w2.updateDev();
	init_zero(dw2);

	init_zero(b2);
	b2.updateDev();
	init_zero(db2);
	
	init_zero(y);
	init_zero(dy);

	init_zero(y_test);

	//double lossval;

	//時間計測開始
	auto start = chrono::system_clock::now();

	cout << "test accuracy in ..." << endl;

	//#pragma acc data copyin(test_data.val[0:test_data.h*test_data.w], w1.val[0:w1.h*w1.w], w2.val[0:w2.h*w2.w], b1.val[0:b1.h*b1.w], b2.val[0:b2.h*b2.w])
	//#pragma acc data create(x.val[0:x.h*x.w], t.val[0:t.h*t.w], dw1.val[0:dw1.h*dw1.w], db1.val[0:db1.h*db1.w], z1.val[0:z1.h*z1.w], dz1.val[0:dz1.h*dz1.w], z1_test.val[0:z1_test.h*z1_test.w], dw2.val[0:dw2.h*dw2.w], db2.val[0:db2.h*db2.w], y.val[0:y.h*y.w], dy.val[0:dy.h*dy.w], y_test.val[0:y_test.h*y_test.w])

	for (ll i=0; i < iters_num; i++) {
		batch_random_choice(train_data, train_label, x, t);
		x.updateDev();
		t.updateDev();
		//順伝播開始
		affine_layer(x, w1, b1, z1);
		//dot(x, w1, z1, NandN);

		//add(z1, b1);
		sigmoid(z1);
		affine_layer(z1, w2, b2, y);
		//dot(z1, w2, y, NandN);
		//add(y, b2);
		softmax(y);
		//lossval = loss(y, t);
		//順伝播終了

		if (i % iter_per_epoch == 0) {
			affine_layer(test_data, w1, b1, z1_test);
			//dot(test_data, w1, z1_test, NandN);
			//add(z1_test, b1);
			sigmoid(z1_test);
			affine_layer(z1_test, w2, b2, y_test);
			//dot(z1_test, w2, y_test, NandN);
			//add(y_test, b2);
			double acc = accuracy(y_test, test_label);
			cout << "iter " << i << " : " << acc << endl;
		}

		//逆伝播開始
		scale_sub(y, t, dy, 1.);

		div_by_scalar(dy, (double)batch_size);

		//softmax with loss 通過
		sum_vertical(dy, db2); //db2求める

		dot(z1, dy, dw2, TandN); //dw2求める
		dot(dy, w2, dz1, NandT); //dz1求める.sigmoidより右

		back_sigmoid(dz1, z1); //sigmoidを後方通過
		sum_vertical(dz1, db1); //db1求める
		dot(x, dz1, dw1, TandN); //dw2求める

		scale_sub(b2, db2, b2, learning_rate); //b2更新
		scale_sub(z1, dz1, z1, learning_rate); //z1更新
		scale_sub(w1, dw1, w1, learning_rate); //w1更新
		scale_sub(b1, db1, b1, learning_rate); //b1更新
		scale_sub(w2, dw2, w2, learning_rate); //w2更新
		//逆伝播終了
	}
	auto end = chrono::system_clock::now();
	auto dur = end - start;
	auto msec = chrono::duration_cast<chrono::milliseconds>(dur).count();

	cout << (double)msec / 1000 << "sec." << endl;


	return 0;
}

void sigmoid(Tensor &a) {
	#pragma acc kernels present(a)
	#pragma acc loop independent gang
	for (ll i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
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

	images.SetDim(number_of_images, rows*cols);

	//cout << magic_number << " " << number_of_images << " " << rows << " " << cols << endl;

	for(int32_t i = 0; i < number_of_images; i++){
		//images[i].resize(rows * cols);

		for(int32_t row = 0; row < rows; row++){
			for(int32_t col = 0; col < cols; col++){
				unsigned char temp = 0;
				ifs.read((char*)&temp,sizeof(temp));
				images[i*rows*cols + row*cols + col] = (double)temp;
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

	label.SetDim(number_of_images, 10);

	//cout << number_of_images << endl;

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
	int32_t a_size = a.h*a.w;
	int32_t b_size = b.h*b.w;
	int32_t c_size = c.h*c.w;
	//double *A, *B, *C;
	//A = a;
	//B = b;
	//C = c;

	//a.updateDev();
	//b.updateDev();

	switch(TorN) {
		case NandN:
			//#pragma acc data copyin(a[0:a_size], b[0:b.h*b.w]) copyout(c[0:c.h*c.w])
			//#pragma acc data copyin(A[0:a_size], B[0:b.h*b.w]) copyout(C[0:c.h*c.w])
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (ll i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (ll j=0; j < n; j++) {
					c[i*n+j] = 0.;
					//C[i*n+j] = 0.;
					#pragma acc loop seq
					for (ll x=0; x < k; x++) {
						c[i*n+j] += a[i*k+x] * b[x*n+j];
					}
				}
			}
			break;
		case TandN:
			//#pragma acc data copyin(a[0:a_size], b[0:b.h*b.w]) copyout(c[0:c.h*c.w])
			//#pragma acc data copyin(A[0:a_size], B[0:b.h*b.w]) copyout(C[0:c.h*c.w])
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (ll i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (ll j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (ll x=0; x < k; x++) {
						c[i*n+j] += a[m*x+i] * b[x*n+j];
					}
				}
			}
			break;
		case NandT:
			//#pragma acc data copyin(a[0:a_size], b[0:b.h*b.w]) copyout(c[0:c.h*c.w])
			//#pragma acc data copyin(A[0:a.h*a.w], B[0:b.h*b.w]) copyout(C[0:c.h*c.w])
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (ll i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (ll j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (ll x=0; x < k; x++) {
						c[i*n+j] += a[i*k+x] * b[k*j+x];
					}
				}
			}
			break;
		case TandT:
			//#pragma acc data copyin(a[0:a_size], b[0:b.h*b.w]) copyout(c[0:c.h*c.w])
			//#pragma acc data copyin(A[0:a.h*a.w], B[0:b.h*b.w]) copyout(C[0:c.h*c.w])
			#pragma acc kernels present(a, b, c)
			#pragma acc loop independent gang
			for (ll i=0; i < m; i++) {
				#pragma acc loop independent vector
				for (ll j=0; j < n; j++) {
					c[i*n+j] = 0.;
					#pragma acc loop seq
					for (ll x=0; x < k; x++) {
						c[i*n+j] += a[m*x+i] * b[k*j+x];
					}
				}
			}
			break;
	}
	//c.updateHost();
}

//行列aの各行にベクトルbを足しこむ
void add_bias(Tensor &a, Tensor &b) {
	if (a.w != b.w) {
		cout << "Tensor size mismatch in add." << endl;
		return;
	}
	#pragma acc kernels present(a, b)
	#pragma acc loop independent gang
	for (ll i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
			a[i*a.w + j] += b[j];
		}
	}
}

//a-scale*bをcに代入
void scale_sub(Tensor &a, Tensor &b, Tensor &c, double scale) {
	if (a.h != b.h || a.w != b.w || a.h != c.h || a.w != c.w) {
		cout << "Tensor size mismatch in sub." << endl;
		return;
	}
	#pragma acc kernels present(a, b, c)
	#pragma acc loop independent gang
	for (ll i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
			c[i*a.w + j] = a[i*a.w + j] - scale*b[i*a.w + j];
		}
	}
}

void div_by_scalar(Tensor &a, double d) {
	#pragma acc kernels present(a)
	#pragma acc loop independent gang
	for (ll i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
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
	for (ll i=0; i < l; i++) {
		t[i] = (s[i] < 0.) ? 0. : s[i];
	}
}

void init_random(Tensor &w) {
	double sigma = 0.01;
	double mean = 0.0;
	int32_t height = w.h;
	int32_t width = w.w;
	random_device seed_gen;
	default_random_engine engine(seed_gen());

	normal_distribution<> dist(mean, sigma);

	for (ll i=0; i < height; i++) {
		for (ll j=0; j < width; j++) {
			w[i*width + j] = dist(engine);
		}
	}
}

void init_zero(Tensor &w) {
	int32_t height = w.h;
	int32_t width = w.w;

	for (ll i=0; i < height; i++) {
		for (ll j=0; j < width; j++) {
			w[i*width + j] = 0.;
		}
	}
}


void batch_random_choice(Tensor &dataset, Tensor &labelset, Tensor &x, Tensor &t) {
	random_device rnd;
	mt19937 mt(rnd());
	uniform_int_distribution<> randbatch(0, dataset.h-1);
	init_zero(t);
	for (ll i=0; i < batch_size; i++) {
		int32_t img_idx = randbatch(mt);
		//cout << img_idx << endl;

		for (ll j=0; j < 10; j++) {
			t[i*10 + j] = labelset[img_idx*10 + j];
		}
		for (ll j=0; j < image_size; j++) {
			x[i*image_size + j] = dataset[img_idx*image_size + j];
		}
	}
}

void softmax(Tensor &a) {
	#pragma acc kernels present(a)
	#pragma acc loop independent gang
	for (ll i=0; i < a.h; i++) {
		double max_pxl = -1000000.;
		#pragma acc loop seq
		for (ll j=0; j < a.w; j++) {
			max_pxl = max(max_pxl, a[i*a.w + j]);
		}
		double sum_exp = 0.;

		double exp_a_c;
		#pragma acc loop seq
		for (ll j=0; j < a.w; j++) {
			exp_a_c = exp(a[i*a.w + j] - max_pxl);
			a[i*a.w + j] = exp_a_c;
			sum_exp += exp_a_c;
		}
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
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

	for (ll i=0; i < y.h; i++) {
		for (ll j=0; j < y.w; j++) {
			los += -t[i*y.w + j] * log(y[i*y.w + j] + delta);
		}
	}
	los /= batch_size;
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
	for (ll i=0; i < a.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < a.w; j++) {
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
	for (ll i=0; i < dz.h; i++) {
		#pragma acc loop independent vector
		for (ll j=0; j < dz.w; j++) {
			dz[i*dz.w+j] *= z[i*dz.w+j] * (1 - z[i*dz.w+j]);
		}
	}
}

double accuracy(Tensor &y, Tensor &t) {
	y.updateHost();
	if(y.h != t.h || y.w != t.w) {
		cout << "Tensor size mismatch in accuracy." << endl;
		return -1.;
	}
	int32_t ymax_idx;
	int32_t tmax_idx;
	double ymax;
	double tmax;
	double acc = 0.;

	for (ll i=0; i < y.h; i++) {
		ymax_idx = -1;
		tmax_idx = -1;
		ymax = -1.;
		tmax = -1.;
		for (ll j=0; j < y.w; j++) {
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
