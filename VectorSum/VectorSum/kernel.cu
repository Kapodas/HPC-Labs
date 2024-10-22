#include "includes.cuh"

using namespace std;

cudaError_t addWithCuda(vector<double>& c, const vector<double>& a, const vector<double>& b, const size_t size);

__global__ void addKernel(double* c, const double* a, const double* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    fillVector fv;
    VectorSum vecSum;
    size_t size;
    cin >> size;

    vector<double> vecA(size);
    vector<double> vecB(size);
    vector<double> vecC;
    vector<double> vecD;
    fv.fillRandom(vecA);
    fv.fillRandom(vecB);
    auto start = chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = addWithCuda(vecD, vecA, vecB, size);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cout << "GPU sum execution time: " << elapsed.count() << " ms" << endl;
    try {
        start = chrono::high_resolution_clock::now();
        vecC = vecSum.cpuSum(vecA, vecB);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
    }
    catch (const invalid_argument& e) {
        cerr << e.what();
    }
    cout << "CPU sum execution time: " << elapsed.count() << " ms" << endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t addWithCuda(vector<double>& c, const vector<double>& a, const vector<double>& b, const size_t size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int blockSize = 1024; // Максимальное количество потоков в блоке
    int numBlocks = (size + blockSize - 1) / blockSize;
    addKernel <<< numBlocks, blockSize >>> (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    c.resize(size);
    cudaStatus = cudaMemcpy(c.data(), dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
