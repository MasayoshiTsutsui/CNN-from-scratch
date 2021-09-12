#pragma once

using ll = int64_t;
using namespace std;

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
};

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
void affine_layer(Tensor &x, Tensor &weight, Tensor &bias, Tensor &z);