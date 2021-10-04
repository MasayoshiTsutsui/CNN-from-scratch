#pragma once

void im2col(Tensor &image, Tensor &expanded, int32_t padsize, int32_t filtersize, int32_t stride);
void col2im(Tensor &expanded, Tensor &image);
void im2col_inverse(Tensor &image, Tensor &expanded, int32_t padsize, int32_t filtersize, int32_t stride);
void col2im_inverse(Tensor &expanded, Tensor &image);
void im2col_pool(Tensor &image, Tensor &expanded, int32_t filtersize);
void col2im_pool(Tensor &expanded, Tensor &image, int32_t filtersize);
void pooling(Tensor &expanded, Tensor &pooled, IntTensor &pooled_idx, int32_t filtersize);
void back_pooling(Tensor &d_before_pool, Tensor &d_pooled, IntTensor &pooled_idx, int32_t filtersize);