#pragma once

void im2col(Tensor &image, Tensor &expanded, int32_t padsize, int32_t filtersize, int32_t stride);
void col2im(Tensor &expanded, Tensor &image);