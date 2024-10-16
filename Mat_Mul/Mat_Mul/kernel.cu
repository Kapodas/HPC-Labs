
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fillMatrix.cu";
#include "MatMul.cu";
#include <stdio.h>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE  16          // submatrix size
#define N           16        // matrix size is N*N

__global__ void matMult(float* a, float* b, int n, float* c)
{
    int   bx = blockIdx.x;     // block index
    int   by = blockIdx.y;
    int   tx = threadIdx.x;        // thread index
    int   ty = threadIdx.y;
    float sum = 0.0f;           // computed subelement
    int   ia = n * BLOCK_SIZE * by + n * ty;   // a [i][0]
    int   ib = BLOCK_SIZE * bx + tx;

    // Multiply the two matrices together;
    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    c[ic + n * ty + tx] = sum;
}

int main(int argc, char* argv[])
{
    int numBytes = N * N * sizeof(float);

    // allocate host memory
    float* a = new float[N * N];
    float* b = new float[N * N];
    float* c = new float[N * N];
    float* d = new float[N * N];
    fillMatrix filler;
    filler.fillRandom(a, N);
    filler.fillRandom(b, N);
    MatMul mul;
    // allocate device memory
    float* adev = NULL;
    float* bdev = NULL;
    float* cdev = NULL;

    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    // set kernel launch configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    auto gpuStart = chrono::high_resolution_clock::now();
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    matMult <<< blocks, threads >>> (adev, bdev, N, cdev);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
    auto gpuEnd = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> gpuTime = gpuEnd - gpuStart;

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);
    // release resources

    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);

    auto cpuStart = chrono::high_resolution_clock::now();
    mul.cpuMul(a, b, d, N);
    auto cpuEnd = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpuTime = cpuEnd - cpuStart;
    printf("time spent executing by the CPU: %.2f millseconds\n", cpuTime);

    delete a;
    delete b;
    delete c;
    delete d;
    return 0;
}