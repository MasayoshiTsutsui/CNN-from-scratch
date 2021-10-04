#include <iostream>
#include "tensor.hpp"

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

void im2col_inverse(Tensor &image, Tensor &expanded, int32_t padsize, int32_t filtersize, int32_t stride) {
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
								//expanded[exp_idx] = 0.;
							}
							else {
								image[d*imgsize3 + c*imgsize2 + h_Img*width + w_Img] = expanded[exp_idx];
							}
							exp_idx++;
						}
					}
				}
			}
		}
	}
}

void col2im(Tensor &expanded, Tensor &image) {
	int32_t datanum = image.d;
	int32_t channel = image.c;
	int32_t height = image.h;
	int32_t width = image.w;

	if(expanded.d != 1 || expanded.c != 1 || expanded.size != image.size) {
		cout << "Tensor size mismatch in img2col" << endl;
		return;
	}

	int32_t imgsize3 = image.size / datanum;
	int32_t imgsize2 = imgsize3 / channel;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				for (int32_t w=0; w < width; w++) {
					image[d*imgsize3 + c*imgsize2 + h*width + w] = expanded[d*imgsize3 + (h*width + w)*channel + c];
				}
			}
		}
	}
}

void col2im_inverse(Tensor &expanded, Tensor &image) {
	int32_t datanum = image.d;
	int32_t channel = image.c;
	int32_t height = image.h;
	int32_t width = image.w;

	if(expanded.d != 1 || expanded.c != 1 || expanded.size != image.size) {
		cout << "Tensor size mismatch in img2col" << endl;
		return;
	}

	int32_t imgsize3 = image.size / datanum;
	int32_t imgsize2 = imgsize3 / channel;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				for (int32_t w=0; w < width; w++) {
					expanded[d*imgsize3 + (h*width + w)*channel + c] = image[d*imgsize3 + c*imgsize2 + h*width + w];
				}
			}
		}
	}
}


//pooling用im2col.
//4次元データを、d,c,wの3次元に展開
void im2col_pool(Tensor &image, Tensor &expanded, int32_t filtersize) {
	int32_t datanum = image.d;
	int32_t channel = image.c;
	int32_t height = image.h;
	int32_t width = image.w;


	if(image.size != expanded.size) {
		cout << "Tensor size mismatch in img2col_pool" << endl;
		return;
	}

	int32_t exp_idx = 0;
	int32_t imgsize3 = image.size / datanum;
	int32_t imgsize2 = imgsize3 / channel;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t ih=0; ih <= height-filtersize; ih+=filtersize) { //filterに取り込まれる最左上の要素のy座標
				for (int32_t iw=0; iw <= width-filtersize; iw+=filtersize) { //同上のx座標
					for (int32_t fh=0; fh < filtersize; fh++) {
						for (int32_t fw=0; fw < filtersize; fw++) {
							int32_t h_Img = ih + fh;
							int32_t w_Img = iw + fw;
							expanded[exp_idx] = image[d*imgsize3 + c*imgsize2 + h_Img*width + w_Img];
							exp_idx++;
						}
					}
				}
			}
		}
	}
}

void col2im_pool(Tensor &expanded, Tensor &image, int32_t filtersize) {
	int32_t datanum = image.d;
	int32_t channel = image.c;
	int32_t height = image.h;
	int32_t width = image.w;


	if(image.size != expanded.size) {
		cout << "Tensor size mismatch in img2col_pool" << endl;
		return;
	}

	int32_t exp_idx = 0;
	int32_t imgsize3 = image.size / datanum;
	int32_t imgsize2 = imgsize3 / channel;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t ih=0; ih <= height-filtersize; ih+=filtersize) { //filterに取り込まれる最左上の要素のy座標
				for (int32_t iw=0; iw <= width-filtersize; iw+=filtersize) { //同上のx座標
					for (int32_t fh=0; fh < filtersize; fh++) {
						for (int32_t fw=0; fw < filtersize; fw++) {
							int32_t h_Img = ih + fh;
							int32_t w_Img = iw + fw;
							image[d*imgsize3 + c*imgsize2 + h_Img*width + w_Img] = expanded[exp_idx];
							exp_idx++;
						}
					}
				}
			}
		}
	}
}

void back_pooling(Tensor &d_before_pool, Tensor &d_pooled, IntTensor &pooled_idx, int32_t filtersize) {
	
	if (d_before_pool.d != d_pooled.d || d_pooled.d != pooled_idx.d || d_before_pool.c != d_pooled.c || d_pooled.c != pooled_idx.c || d_before_pool.h != d_pooled.h || d_pooled.h != pooled_idx.h || d_before_pool.w != filtersize*filtersize || d_pooled.w != 1 || pooled_idx.w != 1) {
		cout << "Tensor size mismatch in back_pooling" << endl;
		return;
	}

	int32_t datanum = d_before_pool.d;
	int32_t channel = d_before_pool.c;
	int32_t height = d_before_pool.h; //widthだけはd_before_poolとpooledで異なる

	int32_t bpsize3 = d_before_pool.size / d_before_pool.d;
	int32_t bpsize2 = bpsize3 / d_before_pool.c;

	int32_t poolsize3 = d_pooled.size / d_pooled.d;
	int32_t poolsize2 = poolsize3 / d_pooled.c;

	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				int32_t max_idx = pooled_idx[d*channel*height + c*height + h];
				for (int32_t w=0; w < d_before_pool.w; w++) {
					if (w == max_idx) {
						d_before_pool[d*bpsize3 + c*bpsize2 + h*d_before_pool.w + w] = d_pooled[d*channel*height + c*height + h];
					}
					else {
						d_before_pool[d*bpsize3 + c*bpsize2 + h*d_before_pool.w + w] = 0.;
					}
				}
			}
		}
	}

}

//expandedのw方向のmaxを取ってpooledにする
void pooling(Tensor &expanded, Tensor &pooled, IntTensor &pooled_idx, int32_t filtersize) {
	if (expanded.d != pooled.d || pooled.d != pooled_idx.d || expanded.c != pooled.c || pooled.c != pooled_idx.c || expanded.h != pooled.h || pooled.h != pooled_idx.h || expanded.w != filtersize*filtersize || pooled.w != 1 || pooled_idx.w != 1) {
		cout << "Tensor size mismatch in pooling" << endl;
		return;
	}
	int32_t datanum = expanded.d;
	int32_t channel = expanded.c;
	int32_t height = expanded.h; //widthだけはexpandedとpooledで異なる

	int32_t expsize3 = expanded.size / expanded.d;
	int32_t expsize2 = expsize3 / expanded.c;

	int32_t poolsize3 = pooled.size / pooled.d;
	int32_t poolsize2 = poolsize3 / pooled.c;


	for (int32_t d=0; d < datanum; d++) {
		for (int32_t c=0; c < channel; c++) {
			for (int32_t h=0; h < height; h++) {
				float max = -9999.;
				int32_t max_idx = -1;
				for (int32_t w=0; w < expanded.w; w++) {
					if (max < expanded[d*expsize3 + c*expsize2 + h*expanded.w + w]) {
						max = expanded[d*expsize3 + c*expsize2 + h*expanded.w + w];
						max_idx = w;
					}
				}
				if (max_idx == -1) {
					cout << "error in pooling!" << endl;
					return;
				}
				pooled[d*poolsize3 + c*poolsize2 + h] = max;
				pooled_idx[d*poolsize3 + c*poolsize2 + h] = max_idx; //maxとして選ばれたもののw方向のidxを記録
			}
		}
	}
}


//int main(){
	//int32_t datanum = 2;
	//int32_t channel = 3;
	//int32_t width = 3;
	//int32_t height = 3;
	//int32_t padsize = 1;
	//int32_t filtersize = 3;
	//int32_t stride = 1;
	//Tensor a(datanum, channel, height, width);
	
	//for (int32_t i=0; i < a.size; i++) {
		//a[i] = (float)i;
	//}
	////a.Print();
	//cout << endl;


	//Tensor b(1, 1, ((width+padsize*2-filtersize+1) / stride)*((height+padsize*2-filtersize+1) / stride)*datanum, filtersize*filtersize*channel);

	//im2col(a, b, padsize, filtersize, stride);

	////b.Print();

	//Tensor c(1,1,datanum*height*width,channel);
	//for (int32_t i=0; i < c.size; i++) {
		//c[i] = (float)i;
	//}

	//Tensor d(datanum, channel, height, width);

	//col2im(c, d);
	//d.Print();
	
	//return 0;
//}