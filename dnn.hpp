#pragma once

uint32_t reverseInt (uint32_t i);
void sigmoid(Tensor &a);
void relu(Tensor &s, IntTensor &erasedmask);
void back_relu(Tensor &ds, IntTensor &erasedmask);
void dot(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
void add_bias(Tensor &a, Tensor &b); //aは行列、bはベクトル。aの各行にbを足しこむ
void scale_sub(Tensor &a, Tensor &b, Tensor &c, float scale); //a-scale*b
uint32_t reverseInt (uint32_t i);
void readTrainingFile(string filename, Tensor &images);
void readLabelFile(string filename, Tensor &label);
void init_random(Tensor &w);
void init_zero(Tensor &w);
void init_zeroint(IntTensor &w);
void batch_random_choice(Tensor &dataset, Tensor &labelset, Tensor &x, Tensor &t);
void softmax(Tensor &a);
double loss(Tensor &y, Tensor &t);
void div_by_scalar(Tensor &a, float d);
void sum_vertical(Tensor &a, Tensor &v);
void back_sigmoid(Tensor &dz, Tensor &z);
float accuracy(Tensor &y, Tensor &t);
void affine_layer(Tensor &x, Tensor &weight, Tensor &bias, Tensor &z);
void dotTC(Tensor &a, Tensor &b, Tensor &c, int32_t TorN);
