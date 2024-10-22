#include <iostream>
#include <random>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

using namespace ::std;

class fillVector {
public:
    void fillRandom(vector<double>& vectr) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);
        for (auto& elem : vectr) {
            elem = dis(gen);
        }
    }
};