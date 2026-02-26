#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attn_kernel(const float* q, const float* k, const float* v, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = q[idx] * k[idx] + v[idx]; 
    }
}

torch::Tensor flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto out = torch::zeros_like(q);
    int threads = 256;
    int blocks = (q.numel() + threads - 1) / threads;
    flash_attn_kernel<<<blocks, threads>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), q.numel());
    return out;
}