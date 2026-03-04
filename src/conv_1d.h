//
// Created by dran on 2/4/26.
//

#ifndef CONV_KERNEL_CONV_1D_H
#define CONV_KERNEL_CONV_1D_H


// This function takes two 1d array of size n and convolve them
// then stores the result in r, which has 2d space allocated
void conv_1d_naive(float* a, float* b, float* c, int n);

// This is the same algorithm that has the innter loop shifted to make 
// one square iteration space
void conv_1d_loop_shift(float* a, float* b, float* c, int n);

// This is the above function with tiling applied to better utilized cache
void conv_1d_loop_shift_tiled(float* a, float* b, float* c, int n);

// This recursive, divide and conqur algortihm implments the same convolution
// using the Karatsuba multiplication algorithm
// TODO : add algorithm design explainations, specifically about the temp array allocation here
void conv_1d_karatsuba(float* a, float* b, float* c, int n);


#endif //CONV_KERNEL_CONV_1D_H
