#include <iostream>
#include <random>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

using namespace ::std;

class fillMatrix {
public:
    void fillRandom(float* matrix, size_t N) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);
        for (size_t i = 0; i < N * N; ++i) {
            matrix[i] = static_cast<float>(dis(gen));
        }
    }
};