import torch
import time
import flash_attn_cuda

N = 1024 * 1024
q, k, v = [torch.randn(N, device='cuda', dtype=torch.float32) for _ in range(3)]

# Benchmark Custom Kernel
torch.cuda.synchronize()
start = time.time()
out_flash = flash_attn_cuda.flash_attn(q, k, v)
torch.cuda.synchronize()
flash_time = time.time() - start

# Benchmark Native
torch.cuda.synchronize()
start = time.time()
out_native = q * k + v
torch.cuda.synchronize()
native_time = time.time() - start

print(f"Fused Kernel: {flash_time:.6f} s")
print(f"PyTorch Native: {native_time:.6f} s")
print(f"Speedup: {native_time / flash_time:.2f}x")