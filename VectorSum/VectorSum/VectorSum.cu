#include <iostream>
#include <random>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

using namespace ::std;

class VectorSum {
private:
    bool canSum(int sizeA, int sizeB) {
        return sizeA == sizeB;
    }
public:
    vector<double> cpuSum(vector<double> A, vector<double> B) {
        if (!canSum(A.size(), B.size())) throw invalid_argument("Error");
        vector<double> C(A.size());
        for (int i = 0; i < C.size(); i++) {
            C[i] = A[i] + B[i];
        }
        return C;
    }
};
