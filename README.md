<h1 align="center"> CUDA Kernel Fusion Engine</h1>
<p align="center">
High-Performance GPU Optimization using PyTorch C++ Extensions
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11.8-green?style=for-the-badge&logo=nvidia">
  <img src="https://img.shields.io/badge/C++-17-blue?style=for-the-badge&logo=c%2B%2B">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-ee4c2c?style=for-the-badge&logo=pytorch">
</p>

##  Problem Statement
Standard PyTorch eager execution performs operations sequentially. For an operation like `out = q * k + v`, PyTorch:
1. Launches a kernel for `q * k`, storing a temporary result in VRAM.
2. Launches a second kernel to add `v` to that temporary result.
3. This creates **redundant memory bandwidth overhead** and multiple kernel launch latencies.

**Goal:** Reduce memory traffic by fusing these steps into a single custom CUDA kernel call.

##  Overview
This project demonstrates how replacing eager GPU operations with a **single custom fused CUDA kernel** reduces memory round-trips. By keeping data in high-speed registers instead of writing intermediate results back to Global Memory (VRAM), we achieve significant speedups. 

**This mirrors real-world kernel fusion strategies used in production-grade frameworks like TensorRT and XLA.**




##  Technical Architecture & Pipeline
The project implements a full-stack integration:
1. **Python Layer**: High-level tensor API for seamless integration.
2. **C++ Binding**: PyBind11 dispatcher for efficient CPU-to-GPU handoff.
3. **CUDA Kernel**: Hand-written `__global__` function for element-wise fusion.
4. **GPU SMs**: Optimized execution across CUDA Streaming Multiprocessors.

<p align="center">
  <img width="1536" height="1024" alt="architechure_daigram" src="https://github.com/user-attachments/assets/2cdb8db4-5f74-4a2c-829b-c13239ae2f64" width="800">
</p>

##  Why Not `torch.compile`?
While PyTorch 2.x offers automatic fusion via `torch.compile`, this project focuses on **manual kernel engineering**. Custom kernels allow for fine-grained control over:
- **Register Pressure**: Managing how many registers each thread uses to maintain high occupancy.
- **Memory Coalescing**: Ensuring global memory accesses are grouped into single transactions.
- **Environment Constraints**: Optimization in systems where a full JIT compiler stack like OpenAI Triton is unavailable.

##  Performance Benchmarks
**Environment:** NVIDIA T4 GPU | CUDA 11.8 | Compute Capability 7.5

| Implementation | Latency (ms) | Speedup |
| :--- | :--- | :--- |
| PyTorch Native (Eager) | 13.46 ms | 1.00× |
| **Fused CUDA Kernel** | **6.69 ms** | **2.01×** |

> **Analysis**: The 2.01x speedup is mathematically defensible: we reduced Global Memory (VRAM) Load/Store operations from 3 passes (Read Q, Read K, Write Temp; Read Temp, Read V, Write Out) down to 1 pass.

##  Future Improvements
- **Shared Memory Tiling**: Implementing tile-based caching for matrix-style operations.
- **Warp-level Primitives**: Using `__shfl_sync__` for faster intra-warp data exchange.
- **Mixed Precision**: Optimizing for Tensor Cores using FP16/BF16.

##  Build & Run
```bash
# Install requirements
pip install -r requirements.txt

# Build the C++ Extension
python setup.py install

# Run performance benchmark
python scripts/benchmark.py
```
---
<p align="center">
  <i>Developed by Manoj Anandan</i>
</p>
