#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;
using ll = int64_t;

int32_t batch_size = 100;
int32_t feature_size1 = 100;
int32_t iters_num = 10000;
double learning_rate = 0.1;

int32_t image_size;
int32_t label_size;

#define NandN 0
#define TandN 1
#define NandT 2
#define TandT 3


class Tensor
{
	public:
		double* val;
		int32_t h;
		int32_t w;
		
		Tensor();
		Tensor(int32_t height, int32_t width);
		void Print();
		void SetDim(int32_t height, int32_t width);
		void FreeVal();
};

uint32_t reverseInt (uint32_t i);

void sigmoid(Tensor &a);
void relu(Tensor &s, Tensor &t);
void dot(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
void add(Tensor &a, Tensor &b); //aは行列、bはベクトル。aの各行にbを足しこむ
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



//行列を縦方向に和を取り、ベクトルにする
void sum_vertical(Tensor &a, Tensor &v) {
	if (v.h != 1 || a.w != v.w) {
		cout << "Tensor size mismatch in sum_vertical." << endl;
		return;
	}
	init_zero(v);
	for (ll i=0; i < a.h; i++) {
		for (ll j=0; j < a.w; j++) {
			v.val[j] += a.val[i*a.w + j];
		}
	}
}



//dz行列の各要素に、z、(1-z)の対応する各要素をかけていく
void back_sigmoid(Tensor &dz, Tensor &z) {
	if (dz.h != z.h || dz.w != z.w) {
		cout << "Tensor size mismatch in back_sigmoid." << endl;
		return;
	}

	for (ll i=0; i < dz.h; i++) {
		for (ll j=0; j < dz.w; j++) {
			dz.val[i*dz.w+j] *= z.val[i*dz.w+j] * (1 - z.val[i*dz.w+j]);
		}
	}
}

int main() {

	Tensor train_data, train_label;

	readTrainingFile("./mnist/train-images-idx3-ubyte", train_data);
	readLabelFile("./mnist/train-labels-idx1-ubyte", train_label); //one-hot label

	int32_t train_size = train_data.h;
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
	Tensor w2(feature_size1, label_size);
	Tensor dw2(feature_size1, label_size);

	Tensor b2(1, label_size);
	Tensor db2(1, label_size);

	Tensor y(batch_size, label_size);
	Tensor dy(batch_size, label_size);



	init_zero(x);
	init_zero(t);

	init_random(w1);
	init_zero(dw1);

	init_zero(b1);
	init_zero(db1);

	init_zero(z1);
	init_zero(dz1);

	init_random(w2);
	init_zero(dw2);

	init_zero(b2);
	init_zero(db2);
	
	init_zero(y);
	init_zero(dy);

	double lossval;

	
	for (ll i=0; i < iters_num; i++) {
		batch_random_choice(train_data, train_label, x, t);
		//順伝播開始
		dot(x, w1, z1, NandN);
		add(z1, b1);
		sigmoid(z1);
		dot(z1, w2, y, NandN);
		add(y, b2);
		softmax(y);
		lossval = loss(y, t);
		//順伝播終了

		if (i % 100 == 0) {
			cout << "iter " << i << " : " << lossval << endl;
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
	train_data.FreeVal();
	train_label.FreeVal();
	x.FreeVal();
	t.FreeVal();
	w1.FreeVal();
	dw1.FreeVal();
	b1.FreeVal();
	db1.FreeVal();
	z1.FreeVal();
	dz1.FreeVal();
	w2.FreeVal();
	dw2.FreeVal();
	b2.FreeVal();
	db2.FreeVal();
	y.FreeVal();
	dy.FreeVal();

	return 0;
}

Tensor::Tensor() {
	h = 0; w = 0;
}
Tensor::Tensor(int32_t height, int32_t width) {
	h = height; w = width;
	val = (double*)malloc(sizeof(double) * height * width);
}

void Tensor::Print() {
	for (ll i=0; i < h; i++) {
		for (ll j=0; j < w; j++) {
			cout << val[i*w+j] << " ";
		}
		cout << endl;
	}
}

void Tensor::SetDim(int32_t height, int32_t width) {
	h = height; w = width;
	val = (double*)malloc(sizeof(double) * height * width);
}

void Tensor::FreeVal() {
	free(val);
}

void sigmoid(Tensor &a) {
	for (ll i=0; i < a.h; i++) {
		for (ll j=0; j < a.w; j++) {
			a.val[i*a.w + j] = 1. / (1. + exp(-a.val[i*a.w + j]));
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
		//images.val[i].resize(rows * cols);

		for(int32_t row = 0; row < rows; row++){
			for(int32_t col = 0; col < cols; col++){
				unsigned char temp = 0;
				ifs.read((char*)&temp,sizeof(temp));
				images.val[i*rows*cols + row*cols + col] = (double)temp;
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
		label.val[i*10 + (int32_t)temp] = 1.;
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
	switch(TorN) {
		case NandN:
			for (ll i=0; i < m; i++) {
				for (ll j=0; j < n; j++) {
					c.val[i*n+j] = 0.;
					for (ll x=0; x < k; x++) {
						c.val[i*n+j] += a.val[i*k+x] * b.val[x*n+j];
					}
				}
			}
			break;
		case TandN:
			for (ll i=0; i < m; i++) {
				for (ll j=0; j < n; j++) {
					c.val[i*n+j] = 0.;
					for (ll x=0; x < k; x++) {
						c.val[i*n+j] += a.val[a.w*x+i] * b.val[x*n+j];
					}
				}
			}
			break;
		case NandT:
			for (ll i=0; i < m; i++) {
				for (ll j=0; j < n; j++) {
					c.val[i*n+j] = 0.;
					for (ll x=0; x < k; x++) {
						c.val[i*n+j] += a.val[i*k+x] * b.val[b.w*j+x];
					}
				}
			}
			break;
		case TandT:
			for (ll i=0; i < m; i++) {
				for (ll j=0; j < n; j++) {
					c.val[i*n+j] = 0.;
					for (ll x=0; x < k; x++) {
						c.val[i*n+j] += a.val[a.w*x+i] * b.val[b.w*j+x];
					}
				}
			}
			break;
	}
}

//行列aの各行にベクトルbを足しこむ
void add(Tensor &a, Tensor &b) {
	if (a.w != b.w) {
		cout << "Tensor size mismatch in add." << endl;
		return;
	}
	for (ll i=0; i < a.h; i++) {
		for (ll j=0; j < a.w; j++) {
			a.val[i*a.w + j] += b.val[j];
		}
	}
}

//a-scale*bをcに代入
void scale_sub(Tensor &a, Tensor &b, Tensor &c, double scale) {
	if (a.h != b.h || a.w != b.w || a.h != c.h || a.w != c.w) {
		cout << "Tensor size mismatch in sub." << endl;
		return;
	}
	for (ll i=0; i < a.h; i++) {
		for (ll j=0; j < a.w; j++) {
			c.val[i*a.w + j] = a.val[i*a.w + j] - scale*b.val[i*a.w + j];
		}
	}
}

void div_by_scalar(Tensor &a, double d) {
	for (ll i=0; i < a.h; i++) {
		for (ll j=0; j < a.w; j++) {
			a.val[i*a.w + j] = a.val[i*a.w + j] / d;
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
		t.val[i] = (s.val[i] < 0.) ? 0. : s.val[i];
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
			w.val[i*width + j] = dist(engine);
		}
	}
}

void init_zero(Tensor &w) {
	int32_t height = w.h;
	int32_t width = w.w;

	for (ll i=0; i < height; i++) {
		for (ll j=0; j < width; j++) {
			w.val[i*width + j] = 0.;
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
			t.val[i*10 + j] = labelset.val[img_idx*10 + j];
		}
		for (ll j=0; j < image_size; j++) {
			x.val[i*image_size + j] = dataset.val[img_idx*image_size + j];
		}
	}
}

void softmax(Tensor &a) {
	for (ll i=0; i < a.h; i++) {
		double max_pxl = -1000000.;
		for (ll j=0; j < a.w; j++) {
			if (max_pxl < a.val[i*a.w + j]) {
				max_pxl = a.val[i*a.w + j];
			}
		}
		double sum_exp = 0.;
		double exp_a_c;
		for (ll j=0; j < a.w; j++) {
			exp_a_c = exp(a.val[i*a.w + j] - max_pxl);
			a.val[i*a.w + j] = exp_a_c;
			sum_exp += exp_a_c;
		}
		for (ll j=0; j < a.w; j++) {
			a.val[i*a.w + j] /= sum_exp;
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
	double lossval = 0.;

	for (ll i=0; i < y.h; i++) {
		for (ll j=0; j < y.w; j++) {
			lossval += -t.val[i*y.w + j] * log(y.val[i*y.w + j] + delta);
		}
	}
	lossval /= batch_size;
	return lossval;
}
