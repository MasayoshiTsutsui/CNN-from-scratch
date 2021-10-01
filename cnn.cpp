#include <iostream>
#include <cmath>
#include "dnn.hpp"

using namespace std;

//filtersize : 1channelあたりのフィルターの大きさ(3*3filterなら3)
void im2col(Tensor &image, Tensor &expanded, int32_t padsize, int32_t filtersize, int32_t stride) {
	int32_t datanum = image.d;
	int32_t channel = image.c;
	int32_t height = image.h;
	int32_t width = image.w;
	if(expanded.d != 1 || expanded.c != 1) {
		cout << "Tensor size mismatch in img2col" << endl;
		return;
	}

	int32_t exp_idx = 0;
	int32_t imgsize3 = image.size / datanum;
	int32_t imgsize2 = imgsize3 / channel;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t ih=-padsize; ih <= height+padsize-filtersize; ih+=stride) { //filterに取り込まれる最左上の要素のy座標
			for (int32_t iw=-padsize; iw <= width+padsize-filtersize; iw+=stride) { //同上のx座標
				for (int32_t c=0; c < channel; c++) {
					for (int32_t fh=0; fh < filtersize; fh++) {
						for (int32_t fw=0; fw < filtersize; fw++) {
							int32_t h_Img = ih + fh;
							int32_t w_Img = iw + fw;
							if (h_Img < 0 || w_Img < 0 || h_Img >= height || w_Img >= width) {
								expanded[exp_idx] = 0.;
							}
							else {
								expanded[exp_idx] = image[d*imgsize3 + c*imgsize2 + h_Img*width + w_Img];
							}
							exp_idx++;
						}
					}
				}
			}
		}
	}
}


int main(){
	int32_t datanum = 2;
	int32_t channel = 3;
	int32_t width = 10;
	int32_t height = 10;
	int32_t padsize = 1;
	int32_t filtersize = 3;
	int32_t stride = 1;
	Tensor a(datanum, channel, height, width);
	
	for (int32_t i=0; i < a.size; i++) {
		a[i] = (float)i;
	}
	a.Print();
	cout << endl;


	Tensor b(1, 1, ((width+padsize*2-filtersize+1) / stride)*((height+padsize*2-filtersize+1) / stride)*datanum, filtersize*filtersize*channel);

	im2col(a, b, padsize, filtersize, stride);

	b.Print();
	
	return 0;
}