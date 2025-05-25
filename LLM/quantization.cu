#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define RNG_NEG -128
#define RNG_POS 127

__global__ void quantize_weights_kernel(
    const float* __restrict__ weights,
    int8_t* __restrict__ weights_q,
    const float scale,
    const int num_elements
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float w = weights[idx];
    int q = roundf(w/scale);
    q = max(RNG_NEG, min(q, RNG_POS));
    weights_q[idx] = static_cast<int8_t>(q);
}

void quantize_weights_cuda(
    torch::Tensor weights,
    torch::Tensor weights_q,
    float scale
){
    const int num_elements = weights.numel();
    const int blocks = (num_elements + THREADS_PER_BLOCK -  1) / THREADS_PER_BLOCK;
    quantize_weights_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        weights.data_ptr<float>();
        weights_q.data_ptr<int8_t>();
        scale, 
        num_elements
    );
}

cudaDeviceSynchronize();
