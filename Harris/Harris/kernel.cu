#include "EasyBMP.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <limits>

// Константы для алгоритма Харриса
#define KERNEL_SIZE 3
#define K 0.04f
#define THRESHOLD -219192508410.0f
// Структура для хранения координат угловых точек
struct Corner {
    int x, y;
};

// CUDA ядро для вычисления угловых точек по алгоритму Харриса
__global__ void harrisCornerDetection(float* d_image, float* d_corners, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    float Ix = 0.0f, Iy = 0.0f;
    float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;

    // Вычисление градиентов Ix и Iy
    for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; i++) {
        for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; j++) {
            int x = idx + i;
            int y = idy + j;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                float dx = d_image[y * width + (x + 1)] - d_image[y * width + (x - 1)];
                float dy = d_image[(y + 1) * width + x] - d_image[(y - 1) * width + x];
                Ix += dx;
                Iy += dy;
            }
        }
    }

    // Вычисление элементов матрицы M
    Ixx = Ix * Ix;
    Iyy = Iy * Iy;
    Ixy = Ix * Iy;

    // Вычисление отклика Харриса
    float det = Ixx * Iyy - Ixy * Ixy;
    float trace = Ixx + Iyy;
    float harrisResponse = det - K * trace * trace;

    // Сохранение отклика Харриса
    d_corners[idy * width + idx] = harrisResponse;
}

int main() {
    // Загрузка изображения
    BMP inputImage;
    inputImage.ReadFromFile("input.bmp");

    int width = inputImage.TellWidth();
    int height = inputImage.TellHeight();

    // Преобразование изображения в массив float
    std::vector<float> h_image(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RGBApixel pixel = inputImage.GetPixel(x, y);
            h_image[y * width + x] = 0.299f * pixel.Red + 0.587f * pixel.Green + 0.114f * pixel.Blue;
        }
    }

    // Выделение памяти на GPU
    float* d_image;
    float* d_corners;
    cudaMalloc(&d_image, width * height * sizeof(float));
    cudaMalloc(&d_corners, width * height * sizeof(float));

    // Копирование данных на GPU
    cudaMemcpy(d_image, h_image.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Запуск CUDA ядра
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    harrisCornerDetection << <gridSize, blockSize >> > (d_image, d_corners, width, height);

    // Копирование результатов с GPU
    std::vector<float> h_corners(width * height);
    cudaMemcpy(h_corners.data(), d_corners, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Поиск угловых точек
    std::vector<Corner> corners;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float response = h_corners[y * width + x];
            if (response < THRESHOLD && response > -std::numeric_limits<float>::infinity()) {
                corners.push_back({ x, y });
            }
        }
    }

    // Отметка угловых точек на изображении
    for (const auto& corner : corners) {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int cx = corner.x + i;
                int cy = corner.y + j;
                if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                    RGBApixel redPixel;
                    redPixel.Red = 255;
                    redPixel.Green = 0;
                    redPixel.Blue = 0;

                    inputImage.SetPixel(cx, cy, redPixel);
                }
            }
        }
    }

    // Сохранение результата
    inputImage.WriteToFile("output.bmp");

    // Освобождение памяти
    cudaFree(d_image);
    cudaFree(d_corners);

    return 0;
}