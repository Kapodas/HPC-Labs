#include <iostream>
#include <random>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

using namespace ::std;

class MatMul {
private:
    bool canMul(int colsA, int rowsB) {
        return colsA == rowsB;
    }
public:
    void cpuMul(float* A, float* B, float* C, int N) {
        if (!canMul(N, N)) {
            throw std::invalid_argument("row A != colum B");
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] = 0;
                for (int k = 0; k < N; ++k) {
                    C[i * N + j] += A[i * N + k] * B[k * N + j];
                }
            }
        }
    }
};
