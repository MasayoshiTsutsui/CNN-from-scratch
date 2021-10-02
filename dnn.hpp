#pragma once

using ll = int64_t;
using namespace std;

#define NandN 0
#define TandN 1
#define NandT 2
#define TandT 3

class Tensor
{
	public:
		float* ptr{nullptr};
		int32_t d{0};
		int32_t c{0};
		int32_t h{0};
		int32_t w{0};
		int32_t size{0};

		#pragma acc routine seq
		float& operator[](size_t idx) { return ptr[idx]; };
		
		explicit Tensor() { };
		//コンストラクタでもうgpu側にメモリ領域を確保してしまう
		explicit Tensor(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			d = datanum; c = channel; h = height; w = width;
			size = datanum * channel * height * width;
			ptr = new float[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(ptr[0:size])
		}
		~Tensor() {
			#pragma acc exit data delete(ptr[0:size])
			#pragma acc exit data delete(this)
			delete [] ptr;
			ptr = NULL;
			d = 0; c = 0; h = 0; w = 0;
			size = 0;
		}

		inline void updateHost() {
			#pragma acc update self(ptr[0:size])
		}
		inline void updateDev() {
			#pragma acc update device(ptr[0:size])
		}

		void Print() {
			int32_t idx = 0;

			for (int32_t id=0; id < d; id++) {
				cout << "data" << id << ":" << endl;
				for (int32_t ic=0; ic < c; ic++) {
					cout << "	channel" << ic << ":" << endl;
					for (int32_t ih=0; ih < h; ih++) {
						cout << "		";
						for (int32_t iw=0; iw < w; iw++) {
							cout << ptr[idx] << " ";
							idx++;
						}
						cout << endl;
					}
				}
			}
		}
		void SetDim(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			d = datanum; c = channel; h = height; w = width;
			size = datanum * channel * height * width;
			ptr = new float[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(ptr[0:size])
		}
};

uint32_t reverseInt (uint32_t i);
void sigmoid(Tensor &a);
void relu(Tensor &s, Tensor &t);
void dot(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
void add_bias(Tensor &a, Tensor &b); //aは行列、bはベクトル。aの各行にbを足しこむ
void scale_sub(Tensor &a, Tensor &b, Tensor &c, float scale); //a-scale*b
uint32_t reverseInt (uint32_t i);
void readTrainingFile(string filename, Tensor &images);
void readLabelFile(string filename, Tensor &label);
void init_random(Tensor &w);
void init_zero(Tensor &w);
void batch_random_choice(Tensor &dataset, Tensor &labelset, Tensor &x, Tensor &t);
void softmax(Tensor &a);
double loss(Tensor &y, Tensor &t);
void div_by_scalar(Tensor &a, float d);
void sum_vertical(Tensor &a, Tensor &v);
void back_sigmoid(Tensor &dz, Tensor &z);
float accuracy(Tensor &y, Tensor &t);
void affine_layer(Tensor &x, Tensor &weight, Tensor &bias, Tensor &z);
void dotTC(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
