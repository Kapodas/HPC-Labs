#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_indirect_functions.h>
#include "EasyBMP.h"

#define BLOCK_SIZE 16

__global__ void medianFilter(unsigned char* output, int width, int height, cudaTextureObject_t texObj);

int main() {
    BMP InputImage;
    InputImage.ReadFromFile("Lena.bmp");

    int width = InputImage.TellWidth();
    int height = InputImage.TellHeight();

    unsigned char* h_input = new unsigned char[width * height];
    unsigned char* h_output = new unsigned char[width * height];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            h_input[j * width + i] = InputImage(i, j)->Red; // Assuming grayscale image
        }
    }

    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, h_input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Bind texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    medianFilter << <dimGrid, dimBlock >> > (d_output, width, height, texObj);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Processing time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    BMP OutputImage;
    OutputImage.SetSize(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            RGBApixel pixel;
            pixel.Red = h_output[j * width + i];
            pixel.Green = h_output[j * width + i];
            pixel.Blue = h_output[j * width + i];
            pixel.Alpha = 0;
            OutputImage.SetPixel(i, j, pixel);
        }
    }
    OutputImage.WriteToFile("output.bmp");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeArray(cuArray);
    cudaDestroyTextureObject(texObj);
    delete[] h_input;
    delete[] h_output;

    return 0;
}

__global__ void medianFilter(unsigned char* output, int width, int height, cudaTextureObject_t texObj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned char window[9];
    int idx = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0) nx = 0;
            if (ny < 0) ny = 0;
            if (nx >= width) nx = width - 1;
            if (ny >= height) ny = height - 1;
            window[idx++] = tex2D<unsigned char>(texObj, nx, ny);
        }
    }

    // Sort the window array (using bubble sort for simplicity)
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (window[i] > window[j]) {
                unsigned char temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }

    // Median is the middle element
    output[y * width + x] = window[4];
}