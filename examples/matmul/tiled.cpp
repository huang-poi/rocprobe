#include <hip/hip_runtime.h>
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < (K+TILE-1)/TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row<M && t*TILE+threadIdx.x<K) ? A[row*K+t*TILE+threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y<K && col<N) ? B[(t*TILE+threadIdx.y)*N+col] : 0;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

// perf(examples): add shared memory padding
