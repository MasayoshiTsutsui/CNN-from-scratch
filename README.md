# MNIST Trainer From Scratch on GPU

## About
This is a CNN trainer for MNIST dataset written in C++ and CUDA from scratch.

## Tools for GPU implementation
- CUDA (https://developer.nvidia.com/cuda-toolkit)
- OpenACC (https://www.openacc.org/)

## Requirements
- nvcc
- pgcc
 
## Feature

This program supports NVIDIA Tensor Core(https://www.nvidia.com/en-us/data-center/tensor-cores/). 
Tensor Core is an arithmetic circuit specialized for matrix multiplication operations. 

CUDA can access Tensor Cores through WMMA API like below.

```C++
wmma::fragment<wmma::matrix_a, TILESIZE, TILESIZE, TILESIZE, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, TILESIZE, TILESIZE, TILESIZE, __half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, TILESIZE, TILESIZE, TILESIZE, __half> c_frag;
wmma::fill_fragment(c_frag, __float2half(0.f));

wmma::load_matrix_sync(a_frag, &a_half[wid*ELEMS_TILE], 16);
wmma::load_matrix_sync(b_frag, &b_half[wid*ELEMS_TILE], 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(&c_half[wid*ELEMS_TILE], c_frag, 16, wmma::mem_row_major);
```

Tensor core performs a certain size of matrix multiplications.
We can make large matrix multiplications by splitting each matrix into small 'tiles' and throw it into Tensor Cores.

