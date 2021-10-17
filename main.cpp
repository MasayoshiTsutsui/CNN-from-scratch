#include <iostream>
#include <fstream>
#include <chrono>
#include "tensor.hpp"
#include "dnn.hpp"
#include "cnn.hpp"

void reshape_tensor(Tensor &a, Tensor &b);

using namespace std;

int32_t batch_size = 100;
int32_t feature_size1 = 100;
int32_t iters_num = 1201;
float learning_rate = 0.01;


int32_t image_size;
int32_t image_h;
int32_t image_w;
int32_t image_c;
int32_t label_size;

//畳み込み層パラメータ
int32_t filternum = 30;
int32_t filtersize = 5;
int32_t padsize = 0;
int32_t stride = 1;
//pooling層パラメータ
int32_t poolsize = 2;

#define DEBUG 0

int main() {

	Tensor train_data, train_label;
	Tensor test_data, test_label;

	readTrainingFile("./mnist/train-images-idx3-ubyte", train_data);
	readLabelFile("./mnist/train-labels-idx1-ubyte", train_label); //one-hot label
	readTrainingFile("./mnist/t10k-images-idx3-ubyte", test_data);
	readLabelFile("./mnist/t10k-labels-idx1-ubyte", test_label); //one-hot label
	
	test_data.updateDev();
	test_label.updateDev();

	int32_t train_size = train_data.d;
	int32_t test_size = test_data.d;
	int32_t iter_per_epoch = train_size / batch_size;

	image_h = train_data.h;
	image_w = train_data.w;
	image_c = train_data.c;
	label_size = train_label.w;

	Tensor x(batch_size, image_c, image_h, image_w);
	init_zero(x);
	Tensor t(1, 1, batch_size, label_size);
	init_zero(t);

	int32_t convoluted1_h = (image_h + padsize*2 - filtersize + 1) / stride;
	int32_t convoluted1_w = (image_w + padsize*2 - filtersize + 1) / stride;

	Tensor expanded_x(1, 1, convoluted1_h*convoluted1_w*batch_size, filtersize*filtersize*image_c);
	init_zero(expanded_x);
	expanded_x.updateDev();
	Tensor d_expanded_x(1, 1, expanded_x.h, expanded_x.w);
	init_zero(d_expanded_x);
	d_expanded_x.updateDev();
	Tensor expanded_test_data(1, 1, convoluted1_h*convoluted1_w*test_data.d, filtersize*filtersize*image_c);
	init_zero(expanded_test_data);
	expanded_test_data.updateDev();

	Tensor conv1(1, 1, image_c*filtersize*filtersize, filternum);
	init_random(conv1);
	conv1.updateDev();

	Tensor d_conv1(1, 1, conv1.h, conv1.w);
	init_zero(d_conv1);
	d_conv1.updateDev();

	Tensor raw_convoluted1(1, 1, convoluted1_h*convoluted1_w*batch_size, filternum);
	init_zero(raw_convoluted1);
	raw_convoluted1.updateDev();
	Tensor d_raw_convoluted1(1, 1, raw_convoluted1.h, raw_convoluted1.w);
	init_zero(d_raw_convoluted1);
	d_raw_convoluted1.updateDev();

	Tensor rawconv1_b(1, 1, 1, raw_convoluted1.w); //畳み込み層のbias
	init_zero(rawconv1_b);
	rawconv1_b.updateDev();
	Tensor d_rawconv1_b(1, 1, 1, raw_convoluted1.w); //畳み込み層のbias微小変化
	init_zero(d_rawconv1_b);
	d_rawconv1_b.updateDev();
	IntTensor relu1mask(1, 1, raw_convoluted1.h, raw_convoluted1.w);
	init_zeroint(relu1mask);
	relu1mask.updateDev();



	Tensor raw_convoluted1_test(1, 1, convoluted1_h*convoluted1_w*test_data.d, filternum);
	init_zero(raw_convoluted1_test);
	raw_convoluted1_test.updateDev();
	IntTensor relu1mask_test(1, 1, raw_convoluted1_test.h, raw_convoluted1_test.w);
	init_zeroint(relu1mask_test);
	relu1mask_test.updateDev();

	Tensor reshaped_conv1(batch_size, filternum, convoluted1_h, convoluted1_w); //畳み込みの後、整形
	init_zero(reshaped_conv1);
	reshaped_conv1.updateDev();
	Tensor d_reshaped_conv1(batch_size, filternum, convoluted1_h, convoluted1_w); //畳み込みの後、整形
	init_zero(d_reshaped_conv1);
	d_reshaped_conv1.updateDev();

	Tensor reshaped_conv1_test(test_data.d, filternum, convoluted1_h, convoluted1_w);
	init_zero(reshaped_conv1_test);
	reshaped_conv1_test.updateDev();


	Tensor conv1_for_pool(batch_size, filternum, (convoluted1_h*convoluted1_w)/(poolsize*poolsize) ,poolsize*poolsize); //w方向にpoolingfilter1枚に食われる要素が並ぶ
	init_zero(conv1_for_pool);
	conv1_for_pool.updateDev();

	Tensor d_conv1_for_pool(batch_size, filternum, (convoluted1_h*convoluted1_w)/(poolsize*poolsize) ,poolsize*poolsize); //上記の微小変化
	init_zero(d_conv1_for_pool);
	d_conv1_for_pool.updateDev();
	Tensor conv1_for_pool_test(test_data.d, filternum, (convoluted1_h*convoluted1_w)/(poolsize*poolsize) ,poolsize*poolsize); //w方向にpoolingfilter1枚に食われる要素が並ぶ
	init_zero(conv1_for_pool_test);
	conv1_for_pool_test.updateDev();
	IntTensor pool_selected_idx_test(test_data.d, filternum, (convoluted1_h*convoluted1_w)/(poolsize*poolsize) ,1); //poolingフィルター内でmaxだったもののidxを保持
	init_zeroint(pool_selected_idx_test);
	pool_selected_idx_test.updateDev();

	IntTensor pool_selected_idx(batch_size, filternum, (convoluted1_h*convoluted1_w)/(poolsize*poolsize) ,1); //poolingフィルター内でmaxだったもののidxを保持
	init_zeroint(pool_selected_idx);
	pool_selected_idx.updateDev();

	Tensor pooled_conv1(batch_size, filternum, conv1_for_pool.h, 1);
	init_zero(pooled_conv1);
	pooled_conv1.updateDev();

	Tensor d_pooled_conv1(batch_size, filternum, conv1_for_pool.h, 1);
	init_zero(d_pooled_conv1);
	d_pooled_conv1.updateDev();

	Tensor expanded_pooled_conv1(1, 1, batch_size, filternum * conv1_for_pool.h);
	init_zero(expanded_pooled_conv1);
	expanded_pooled_conv1.updateDev();

	Tensor d_expanded_pooled_conv1(1, 1, batch_size, filternum * conv1_for_pool.h);
	init_zero(d_expanded_pooled_conv1);
	d_expanded_pooled_conv1.updateDev();
	//Tensor reshaped_pool1(batch_size, filternum, convoluted1_h/poolsize, convoluted1_w/poolsize); //poolingの後、整形これいるか？

	Tensor pooled_conv1_test(test_data.d, filternum, conv1_for_pool.h, 1);
	init_zero(pooled_conv1_test);
	pooled_conv1_test.updateDev();

	Tensor expanded_pooled_conv1_test(1, 1, test_data.d, filternum * conv1_for_pool.h);
	init_zero(expanded_pooled_conv1_test);
	expanded_pooled_conv1_test.updateDev();



	//ここからfcn
	Tensor w1(1, 1, expanded_pooled_conv1.w, feature_size1);
	Tensor dw1(1, 1, w1.h, w1.w);
	init_random(w1);
	w1.updateDev();
	init_zero(dw1);
	dw1.updateDev();

	Tensor b1(1, 1, 1, feature_size1);
	Tensor db1(1, 1, 1, feature_size1);
	init_zero(b1);
	b1.updateDev();
	init_zero(db1);
	db1.updateDev();

	Tensor hidden2(1, 1, batch_size, feature_size1);
	Tensor d_hidden2(1, 1, hidden2.h, hidden2.w);
	init_zero(hidden2);
	init_zero(d_hidden2);
	hidden2.updateDev();
	d_hidden2.updateDev();

	IntTensor relu2mask(1, 1, hidden2.h, hidden2.w);
	init_zeroint(relu2mask);
	relu2mask.updateDev();


	Tensor hidden2_test(1, 1, test_size, feature_size1); //1epoch終了時のテスト専用のhidden2
	init_zero(hidden2_test);
	hidden2_test.updateDev();
	IntTensor relu2mask_test(1, 1, hidden2_test.h, hidden2_test.w);
	init_zeroint(relu2mask_test);
	relu2mask_test.updateDev();

	Tensor w2(1, 1, feature_size1, label_size);
	init_random(w2);
	w2.updateDev();
	Tensor dw2(1, 1, feature_size1, label_size);
	init_zero(dw2);
	dw2.updateDev();
	Tensor b2(1, 1, 1, label_size);
	Tensor db2(1, 1, 1, label_size);
	init_zero(b2);
	b2.updateDev();
	init_zero(db2);
	db2.updateDev();

	Tensor y(1, 1, batch_size, label_size);
	Tensor dy(1, 1, batch_size, label_size);

	Tensor y_test(1, 1, test_size, label_size);
	init_zero(y);
	y.updateDev();
	init_zero(dy);
	dy.updateDev();

	init_zero(y_test);
	y_test.updateDev();



	//時間計測開始
	auto start = chrono::system_clock::now();

	cout << "test accuracy in ..." << endl;

	for (int32_t i=0; i < iters_num; i++) {
		batch_random_choice(train_data, train_label, x, t);
		x.updateDev();
		t.updateDev();

		//畳み込み層順伝播開始
		im2col(x, expanded_x, padsize, filtersize, stride);
		dot(expanded_x, conv1, raw_convoluted1, NandN);
		add_bias(raw_convoluted1, rawconv1_b);
		relu(raw_convoluted1, relu1mask);
		col2im(raw_convoluted1, reshaped_conv1);
		im2col_pool(reshaped_conv1, conv1_for_pool, poolsize);
		pooling(conv1_for_pool, pooled_conv1, pool_selected_idx, poolsize);
		reshape_tensor(pooled_conv1, expanded_pooled_conv1);
		//pooled_conv1.Reshape(1, 1, batch_size, pooled_conv1.size / batch_size);

		//順伝播開始
		affine_layer(expanded_pooled_conv1, w1, b1, hidden2);
		relu(hidden2, relu2mask);
		affine_layer(hidden2, w2, b2, y);
		softmax(y);
		//lossval = loss(y, t);
		//順伝播終了

		if (i % 50 == 0) {
			im2col(test_data, expanded_test_data, padsize, filtersize, stride);
			dot(expanded_test_data, conv1, raw_convoluted1_test, NandN);
			add_bias(raw_convoluted1_test, rawconv1_b);
			relu(raw_convoluted1_test, relu1mask_test);
			col2im(raw_convoluted1_test, reshaped_conv1_test);
			im2col_pool(reshaped_conv1_test, conv1_for_pool_test, poolsize);
			//if (i == 40) {
				//cout << "conv1_for_pool_test :" << endl;
				//conv1_for_pool_test.Print();
			//}
			pooling(conv1_for_pool_test, pooled_conv1_test, pool_selected_idx_test, poolsize);
			reshape_tensor(pooled_conv1_test, expanded_pooled_conv1_test);
			//pooled_conv1_test.Reshape(1, 1, test_data.d, pooled_conv1.size / test_data.d);

			affine_layer(expanded_pooled_conv1_test, w1, b1, hidden2_test);
			relu(hidden2_test, relu2mask_test);
			affine_layer(hidden2_test, w2, b2, y_test);
			float acc = accuracy(y_test, test_label);
			cout << "iter " << i << " : " << acc << endl;
		}

		//逆伝播開始
		scale_sub(y, t, dy, 1.);
		div_by_scalar(dy, (float)batch_size);

		//softmax with loss 通過
		sum_vertical(dy, db2); //db2求める

		dot(hidden2, dy, dw2, TandN); //dw2求める
		dot(dy, w2, d_hidden2, NandT); //dz1求める.reluより右

		back_relu(d_hidden2, relu2mask); //reluを後方通過
		sum_vertical(d_hidden2, db1); //db1求める
		dot(expanded_pooled_conv1, d_hidden2, dw1, TandN); //dw1求める
		dot(d_hidden2, w1, d_expanded_pooled_conv1, NandT);
		
		scale_sub(b2, db2, b2, learning_rate); //b2更新
		//scale_sub(hidden2, d_hidden2, hidden2, learning_rate); //z1更新
		scale_sub(w1, dw1, w1, learning_rate); //w1更新
		scale_sub(b1, db1, b1, learning_rate); //b1更新
		scale_sub(w2, dw2, w2, learning_rate); //w2更新

		//fcn層逆伝播通過
		//cnn層逆伝播開始
		reshape_tensor(d_expanded_pooled_conv1, d_pooled_conv1);
		back_pooling(d_conv1_for_pool, d_pooled_conv1, pool_selected_idx, poolsize);
		col2im_pool(d_conv1_for_pool, d_reshaped_conv1, poolsize);
		col2im_inverse(d_raw_convoluted1, d_reshaped_conv1);
		back_relu(d_raw_convoluted1, relu1mask);
		sum_vertical(d_raw_convoluted1, d_rawconv1_b);
		dot(expanded_x, d_raw_convoluted1, d_conv1, TandN);
		//dot(d_raw_convoluted1, conv1, d_expanded_x, NandT);

		scale_sub(rawconv1_b, d_rawconv1_b, rawconv1_b, learning_rate); //畳み込み層b更新
		scale_sub(conv1, d_conv1, conv1, learning_rate); //畳み込み層重み更新

		//逆伝播終了
	}

	auto end = chrono::system_clock::now();
	auto dur = end - start;
	auto msec = chrono::duration_cast<chrono::milliseconds>(dur).count();

	cout << (double)msec / 1000 << "sec." << endl;

	return 0;
	

}


void fcn2layer() {

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
}

void reshape_tensor(Tensor &a, Tensor &b) {
	if (a.size != b.size) {
		cout << "Tensor size mismatch in reshape tensor." << endl;
		return;
	}
	#pragma acc kernels present(a, b)
	#pragma acc loop independent vector
	for (int32_t i=0; i < a.size; i++) {
		b[i] = a[i];
	}
}