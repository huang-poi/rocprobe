#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vec_add(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    size_t n = 1 << 24;
    size_t bytes = n * sizeof(float);
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes); hipMalloc(&d_b, bytes); hipMalloc(&d_c, bytes);
    // ... launch and verify
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    std::cout << "vec_add: " << n << " elements" << std::endl;
}

// perf(examples): optimize with float4 vectorized loads
