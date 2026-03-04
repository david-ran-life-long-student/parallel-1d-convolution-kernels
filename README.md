# parallel-1d-convolution-kernels
CPU compute kernels for demonstrating various parallelization techniques using openMP

A 1D convolution is a fundamental mathematical operation heavily utilized in signal processing, polynomial multiplication, and deep learning. It takes two one-dimensional arrays of length $n$ and produces a third array of length $2n - 1$. The operation involves "sliding" one array over the other, multiplying the overlapping elements, and summing the results. Mathematically, the discrete 1D convolution of two signals $a$ and $b$ is defined as:

$$c[k] = \sum_{j} a[k-j] \cdot b[j]$$


## What's here?

This repository contains four distinct implementations of the 1D convolution, progressively demonstrating different performance and algorithmic optimizations:

* **`conv_1d_naive`**: The standard, direct implementation of the mathematical definition. It handles the sliding window using upper and lower triangular iteration spaces, which naturally results in complex loop bounds and less efficient memory access patterns.
* **`conv_1d_loop_shift`**: An optimized version that shifts the iteration space into a square loop structure. By modifying the access pattern to calculate $a[i] \cdot b[j]$ and accumulate into $c[i+j]$, it eliminates negative strides and makes the loops highly parallelizable.
* **`conv_1d_loop_shift_tiled`**: Builds on the shifted loop design by applying cache blocking. It processes the arrays in configurable chunks (`DEFAULT_LEAF_SIZE`) to ensure the working dataset fits efficiently within the CPU cache.
* **`conv_1d_karatsuba`**: An advanced recursive implementation using the Karatsuba algorithm. This approach achieves sub-quadratic time complexity (roughly $O(n^{1.58})$) by dividing the input arrays into high and low halves, trading a costly recursive multiplication for cheaper vector additions and subtractions.

## parallelization strategies used

This project explores both hardware-focused OpenMP strategies and algorithmic improvements to speed up the convolution kernel:

* **Loop Skewing / Permutation**: Demonstrated in the shifted loop implementations, rewriting the nested loops transforms the triangular iteration space into a rectangular one. This eliminates data dependencies across the inner loop and sets the stage for easy parallelization.
* **Data-Level Parallelism (SIMD)**: Inner loops and inline utility functions (`vec_add`, `vec_sub`, `vec_zero`) utilize the `#pragma omp simd` directive to take advantage of the CPU's vector registers, processing multiple array elements in a single clock cycle.
* **Thread-Level Parallelism with Reductions**: Because the shifted loop approach causes multiple threads to potentially write to the same `c[i+j]` index simultaneously, `#pragma omp parallel for reduction(+:c[:len*2])` is used. This safely parallelizes the outer loop while preventing race conditions on the output array.
* **Cache Tiling (Blocking)**: The tiled implementation collapses the outer loops and breaks memory access into localized blocks. This strategy drastically reduces cache misses.
* **Task-Based Divide and Conquer**: For the Karatsuba algorithm, OpenMP's task-based parallelism (`#pragma omp task` and `#pragma omp taskwait`) is deployed to spawn parallel threads for the recursive, independent multiplications of the array halves.