//
// Created by dran on 2/4/26.
//

#ifndef CONV_KERNEL_CONV_1D_H
#define CONV_KERNEL_CONV_1D_H



void conv_1d_naive(float* a, float* b, float* c, int n);

void conv_1d_loop_shift(float* a, float* b, float* c, int n);

void conv_1d_tiled(float* a, float* b, float* c, int n);

void conv_1d_karatsuba(float* a, float* b, float* c, int n);


#endif //CONV_KERNEL_CONV_1D_H