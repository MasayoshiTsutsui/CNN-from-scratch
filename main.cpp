#include <iostream>
#include <fstream>
#include <chrono>
#include "tensor.hpp"
#include "dnn.hpp"
#include "cnn.hpp"

using namespace std;

int32_t batch_size = 100;
int32_t feature_size1 = 100;
int32_t iters_num = 1201;
float learning_rate = 0.1;


int32_t image_size;
int32_t label_size;

#define NandN 0
#define TandN 1
#define NandT 2
#define TandT 3
#define DEBUG 1


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

	Tensor x(1, 1, batch_size, image_size);
	Tensor t(1, 1, batch_size, label_size);

	Tensor w1(1, 1, image_size, feature_size1);
	Tensor dw1(1, 1, image_size, feature_size1);
	Tensor b1(1, 1, 1, feature_size1);
	Tensor db1(1, 1, 1, feature_size1);
	

	Tensor z1(1, 1, batch_size, feature_size1);
	Tensor dz1(1, 1, batch_size, feature_size1);

	Tensor z1_test(1, 1, test_size, feature_size1);

	Tensor w2(1, 1, feature_size1, label_size);
	Tensor dw2(1, 1, feature_size1, label_size);
	Tensor b2(1, 1, 1, label_size);
	Tensor db2(1, 1, 1, label_size);

	Tensor y(1, 1, batch_size, label_size);
	Tensor dy(1, 1, batch_size, label_size);

	Tensor y_test(1, 1, test_size, label_size);

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


	for (int32_t i=0; i < iters_num; i++) {
		batch_random_choice(train_data, train_label, x, t);
		x.updateDev();
		t.updateDev();
		//順伝播開始
		affine_layer(x, w1, b1, z1);
		sigmoid(z1);
		affine_layer(z1, w2, b2, y);
		softmax(y);
		//lossval = loss(y, t);
		//順伝播終了

		if (i % iter_per_epoch == 0) {
			affine_layer(test_data, w1, b1, z1_test);
			sigmoid(z1_test);
			affine_layer(z1_test, w2, b2, y_test);
			float acc = accuracy(y_test, test_label);
			cout << "iter " << i << " : " << acc << endl;
		}

		//逆伝播開始
		scale_sub(y, t, dy, 1.);

		div_by_scalar(dy, (float)batch_size);

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
