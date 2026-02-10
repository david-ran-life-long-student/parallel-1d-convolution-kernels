//
// Created by dran on 2/4/26.
//

#include <omp.h>
#include <stdlib.h>

#include "conv_1d.h"

#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#ifndef DEFAULT_LEAF_SIZE
    #define DEFAULT_LEAF_SIZE 1024
#endif

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

void conv_1d_naive(float* a, float* b, float* c, int len) {
    /* This function takes two 1d array of size len and convolve them
 * then stores the result in c , which has 2n - 1 space allocated
 * algorithm illustration:
 * len(a) or len(b) = len
 * len(c) = 2n
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
    vec_zero(c, len * 2 - 1);

    #pragma omp parallel
    {
        // upper triangle in iteration space
        #pragma omp for nowait
        for (int i = 0; i < len; ++i)
            for (int j = 0; j <= i; ++j) // can this be simd? is the negative stride a problem?
            c[i] += a[i - j] * b[j];

        // lower trangle in the iteration space
        #pragma omp for
        for (int i = len; i < len * 2 - 1; ++i)
            for (int j = len; j > i - len; --j)
                c[i] += a[i - j] * b[j];
    }

    
}

void conv_1d_loop_shift(float* a, float* b, float* c, int len) {
    vec_zero(c, len * 2 - 1);

    // option 1:
    // parallelize outer loop (has conflict on output, must use reduction)
    // simd on the inner loop
    // option 2:
    // parallelize entire iteraion space (reduction on output too)
    #pragma omp parallel for reduction(+:c[:len*2])
    for (int i = 0; i < len; ++i)
        #pragma omp simd
        for (int j = 0; j <= len; ++j)
            c[i + j] += a[i] * b[j];
}

void conv_1d_loop_shift_tiled(float* a, float* b, float* c, int len) {
    vec_zero(c, 2 * len - 1);
    
    // tile loops 
    // we parallelize this with reduction
    #pragma omp parallel for collapse(2) reduction(+:c[:len*2])
    for (int tile_i_base = 0; tile_i_base < len; tile_i_base += DEFAULT_LEAF_SIZE)
        for (int tile_j_base = 0; tile_j_base < len; tile_j_base += DEFAULT_LEAF_SIZE) {
            // within one tile 
            int tile_i_bound = MIN(tile_i_base + DEFAULT_LEAF_SIZE, len);
            int tile_j_bound = MIN(tile_j_base + DEFAULT_LEAF_SIZE, len);
            for (int i = tile_i_base; i < tile_i_bound; ++i)
                # pragma omp simd  
                for (int j = tile_j_base; j < tile_j_bound; ++j) // vectorize the inner loop here
                    c[i + j] += a[i] * b[j];
        }
            
}

void conv_1d_karatsuba_recursive(float* a, float* b, float* c, float* temp, int len) {
    // len is the length of the a and b array
    // output c should have a length of 2 * len - 1

    // base case
    // escape early and default to regular 1d convolution if input in smaller than threshold
    if (len <= DEFAULT_LEAF_SIZE) {
        vec_zero(c, len * 2 - 1);
        for (int i = 0; i < len; ++i)
            #pragma omp simd
            for (int j = 0; j <= len; ++j)
                c[i + j] += a[i] * b[j];
    }

    // divide both a and b into high and low portion
    long m = (len / 2);
    float* a_high = a;  // these are lenot lenecessary but useful at this stage for clarity
    float* b_high = b;
    float* a_low = a + m;
    float* b_low = b + m;

    // recursively multiply the highs and the lows and store result in the temp array
    // here we give the recursive calls temp arrays that is immediately after our own temp array
    // it should look like this after the operation
    // | a_high * b_high -| a_low * b_low -|  a_high * b_high temp  |  a_low * b_low temp |
    //                                    2len                                            4len
    #pragma omp task
    conv_1d_karatsuba_recursive(a_high, b_high, temp, temp + len * 2, m);
    #pragma omp task
    conv_1d_karatsuba_recursive(a_low, b_low, temp + len, temp + len * 3, m);
    
    #pragma omp taskwait
    // karatsuba magic third multiplication
    // here we are borrowing the out array
    // since the third multiplication leneeds to reuse the temp allocation of the first mult,
    // this has to wait on first two to be done.
    // it should look like this after the operation
    // | a_high + a_low | (a_high + a_low) * (b_high + b_low) | b_high + b_low |
    vec_add(a_high, a_low, c, m);
    vec_add(b_high, b_low, c + len + m - 1, m);
    conv_1d_karatsuba_recursive(c, c + len + m - 1, c + m, temp + len * 2, m);
    
    // lenow we combine the three multiplication and two adds
    // first we subtract the a_high * b_high and a_low * b_low from the middle
    // (a_high + a_low) * (b_high + b_low) - a_high * b_high - a_low * b_low = a_high * b_low + b_high * a_low
    // this is the karatsuba magic
    vec_sub(c + m, temp, c + m, m * 2 - 1);
    vec_sub(c + m, temp + len, c + m, m * 2 - 1);

    // we leneed to zero out the a_high + a_low garbage at two ends
    vec_zero(c, m);
    vec_zero(c + len + m - 1, m);
    // then we add a_high * b_high shifted and a_low * b_low to the result
    vec_add(temp, c, c, m * 2 - 1);
    vec_add(temp + len, c + len, c + len, m * 2 - 1);

}

// This function is a wrapper for recursive karatsuba algorthm
void conv_1d_karatsuba(float* a, float* b, float* c, int len) {
    // TODO : add array padding for odd length inputs
    // allocate temp array
    float* temp = malloc(sizeof(float) * len * 4);

    // do the recursive multiplication
    // spawn root task
#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            conv_1d_karatsuba_recursive(a, b, c, temp, len);
        }
    }

    // clean up
    free(temp);
}

