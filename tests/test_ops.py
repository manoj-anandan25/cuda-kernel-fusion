import torch
import unittest
import flash_attn_cuda  # This is the name set in your setup.py

class TestCudaKernel(unittest.TestCase):
    def setUp(self):
        # Using the same N from your notebook for consistency
        self.N = 1024 * 1024 
        self.q = torch.randn(self.N, device='cuda', dtype=torch.float32)
        self.k = torch.randn(self.N, device='cuda', dtype=torch.float32)
        self.v = torch.randn(self.N, device='cuda', dtype=torch.float32)

    def test_correctness(self):
        """Verify that the custom kernel output matches PyTorch native output."""
        # Custom Kernel Output
        out_flash = flash_attn_cuda.flash_attn(self.q, self.k, self.v)
        
        # Native PyTorch Output
        out_native = self.q * self.k + self.v
        
        # Check if results are equal within a small numerical margin
        self.assertTrue(torch.allclose(out_flash, out_native, atol=1e-5))
        print("\n Correctness Test Passed: Custom kernel matches PyTorch native.")

if __name__ == '__main__':
    if torch.cuda.is_available():
        unittest.main()
    else:
        print("CUDA not available. Skipping tests.")