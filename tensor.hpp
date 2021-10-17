#pragma once

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
		float& operator[](size_t idx) {
			return ptr[idx];
		};

		
		explicit Tensor() { };
		//コンストラクタでもうgpu側にメモリ領域を確保してしまう
		explicit Tensor(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			d = datanum; c = channel; h = height; w = width;
			size = datanum * channel * height * width;
			try {
				ptr = new float[size];
			}
			catch (bad_alloc) {
				cerr << "Memory exhausted" << endl;
			}
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
		void Reshape(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			if (size != datanum*channel*height*width) {
				cout << "Element numnber does not match in Reshape method!" << endl;
				return;
			}
			d = datanum; c = channel; h = height; w = width;
		}
};

class IntTensor
{
	public:
		int32_t* ptr{nullptr};
		int32_t d{0};
		int32_t c{0};
		int32_t h{0};
		int32_t w{0};
		int32_t size{0};

		#pragma acc routine seq
		int32_t& operator[](size_t idx) {
			return ptr[idx];
		};
		
		explicit IntTensor() { };
		//コンストラクタでもうgpu側にメモリ領域を確保してしまう
		explicit IntTensor(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			d = datanum; c = channel; h = height; w = width;
			size = datanum * channel * height * width;
			ptr = new int32_t[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(ptr[0:size])
		}
		~IntTensor() {
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
			ptr = new int32_t[size];
			#pragma acc enter data copyin(this)
			#pragma acc enter data create(ptr[0:size])
		}
		void Reshape(int32_t datanum, int32_t channel, int32_t height, int32_t width) {
			if (size != datanum*channel*height*width) {
				cout << "Element numnber does not match in Reshape method!" << endl;
				return;
			}
			d = datanum; c = channel; h = height; w = width;
		}
};


void init_random(Tensor &w);
void init_zero(Tensor &w);
void init_zeroint(IntTensor &w);