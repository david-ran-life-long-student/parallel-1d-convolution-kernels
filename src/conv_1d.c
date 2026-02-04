//
// Created by dran on 2/4/26.
//

#include <omp.h>

#include "conv_1d.h"

#define DEFAULT_LEAF_SIZE 1024

// some utils
inline void vec_add(const float* a, const float* b, float* out, const int len) {
    #pragma omp simd
    for (long i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

inline void vec_sub(const float* a, const float* b, float* out, const int len) {
    #pragma omp simd
    for (long i = 0; i < len; i++) {
        out[i] = a[i] - b[i];
    }
}

inline void vec_zero(float* a, const int len) {
#pragma omp simd
    for (long i = 0; i < len; i++) {
        a[i] = 0;
    }
}

void conv_1d_naive(float* a, float* b, float* c, int n) {

    /* This function takes two 1d array of size d + 1 and convolve them
 * then stores the result in r, which has 2d space allocated
 * algorithm illustration:
 * len(a) or len(b) = d + 1
 * len(r) = 2d + 1
 * time series  j across i down
 *                      sum
 * 0,0                  r0
 * 1,0  0,1             r1
 * 2,0  1,1  0,2        r2
 * ( j iteration is backwards after this point,
 * but logically it should be like below )
 *      2,1  1,2        r3
 *           2,2        r4
 *
 */
    // zero out return array
    vec_zero(c, 2 * n - 1);

    // upper triangle in iteration space
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            c[i] += a[i - j] * b[j];

    // lower trangle in the iteration space
    for (int i = n; i < n * 2 - 1; ++i)
        for (int j = n; j > i - n; ++j)
            c[i] += a[i - j] * b[j];
    
}

void conv_1d_loop_shift(float* a, float* b, float* c, int n) {
    vec_zero(c, 2 * n - 1);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            c[i + j] += a[i] * b[j];
}

void conv_1d_karatsuba_recursive(float* a, float* b, float* c, float* temp, long len, const long leaf_threshold) {
    // len is the length of the a and b array
    // output c should have a length of 2 * len - 1
    // base case
    // escape early and default to regular 1d convolution if input in smaller than threshold
    if (len <= DEFAULT_LEAF_SIZE)
        return conv_1d_naive(a, b, c, n);


    // divide both a and b into high and low portion
    long m = (len / 2);
    float* a_high = a;  // these are not necessary but useful at this stage for clarity
    float* b_high = b;
    float* a_low = a + m;
    float* b_low = b + m;

    // recursively multiply the highs and the lows and store result in the temp array
    // here we give the recursive calls temp arrays that is immediately after our own temp array
    // it should look like this after the operation
    // | a_high * b_high -| a_low * b_low -|
    #pragma omp task
    conv_1d_karatsuba_recursive(a_high, b_high, temp, temp + len * 2, m, leaf_threshold);
    #pragma omp task
    conv_1d_karatsuba_recursive(a_low, b_low, temp + len, temp + len * 3, m, leaf_threshold);
    
    #pragma omp taskwait
    // karatsuba magic third multiplication
    // here we are borrowing the out array
    // since the third multiplication needs to reuse the temp allocation of the first mult,
    // this has to wait on first two to be done.
    // it should look like this after the operation
    // | a_high + a_low | (a_high + a_low) * (b_high + b_low) | b_high + b_low |
    vec_add(a_high, a_low, c, m);
    vec_add(b_high, b_low, c + len + m - 1, m);
    conv_1d_karatsuba_recursive(c, c + len + m - 1, c + m, temp + len * 2, m, leaf_threshold);
    
    // now we combine the three multiplication and two adds
    // first we subtract the a_high * b_high and a_low * b_low from the middle
    // (a_high + a_low) * (b_high + b_low) - a_high * b_high - a_low * b_low = a_high * b_low + b_high * a_low
    // this is the karatsuba magic
    vec_sub(c + m, temp, c + m, m * 2 - 1);
    vec_sub(c + m, temp + len, c + m, m * 2 - 1);

    // we need to zero out the a_high + a_low garbage at two ends
    vec_zero(c, m);
    vec_zero(c + len + m - 1, m);
    // then we add a_high * b_high shifted and a_low * b_low to the result
    vec_add(temp, c, c, m * 2 - 1);
    vec_add(temp + len, c + len, c + len, m * 2 - 1);

}

