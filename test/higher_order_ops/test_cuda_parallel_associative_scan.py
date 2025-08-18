# mypy: allow-untyped-defs
"""
Tests for CUDA parallel associative scan implementation.

These tests are designed to specifically exercise the CUDA parallel path in
cuda_parallel_associative_scan by ensuring all the required conditions are met:
- combine_mode="pointwise"
- CUDA tensor
- 1D tensor
- Contiguous memory layout
- reverse=False
"""

# This is not used directly here, but removing this results in a build error.
# Something about the import order of cuda.cccl.parallel.experimental and torch
# affects the build.
import cuda.cccl.parallel.experimental as parallel

import math
import unittest
import torch
import io
import sys
from contextlib import redirect_stdout


def skip_if_no_cuda(test_func):
    """Decorator to skip tests if CUDA is not available"""
    def wrapper(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        return test_func(self)
    return wrapper


class TestCudaParallelAssociativeScan(unittest.TestCase):
    """Test cases specifically designed to trigger the CUDA parallel associative scan path."""

    def setUp(self):
        super().setUp()
        # Clear any previous torch compilation state
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()

    def _import_cuda_parallel_scan(self):
        """Import the CUDA parallel scan function, handling potential import errors."""
        try:
            from torch._higher_order_ops.cuda_parallel_associative_scan import cuda_parallel_associative_scan
            return cuda_parallel_associative_scan
        except ImportError as e:
            self.skipTest(f"CUDA parallel scan not available: {e}")

    @skip_if_no_cuda
    def test_cuda_parallel_scan_basic_add(self):
        """Test basic addition with CUDA parallel scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        x = torch.randn(100, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=add_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="pointwise"
            )
        output = f.getvalue()
        print(output)
        
        # Verify the CUDA parallel implementation was used
        self.assertIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness against torch.cumsum
        expected = torch.cumsum(x, dim=0)
        torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_cuda_parallel_scan_basic_multiply(self):
        """Test basic multiplication with CUDA parallel scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def mul_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x * y

        x = torch.randn(50, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=mul_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="pointwise"
            )
        output = f.getvalue()
        
        print(output)
        # Verify the CUDA parallel implementation was used
        self.assertIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness against torch.cumprod
        expected = torch.cumprod(x, dim=0)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @skip_if_no_cuda
    def test_cuda_parallel_scan_different_dtypes(self):
        """Test CUDA parallel scan with different dtypes"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        dtypes_to_test = [torch.float32, torch.float64]
        
        for dtype in dtypes_to_test:
            with self.subTest(dtype=dtype):
                x = torch.randn(75, device="cuda", dtype=dtype).contiguous()
                
                # Capture stdout to verify CUDA parallel scan is used
                f = io.StringIO()
                with redirect_stdout(f):
                    result = cuda_parallel_associative_scan(
                        combine_fn=add_fn,
                        xs=x,
                        dim=0,
                        reverse=False,
                        combine_mode="pointwise"
                    )
                output = f.getvalue()
                
                # Verify the CUDA parallel implementation was used
                self.assertIn("CUDA_PARALLEL_SCAN_USED", output)
                
                # Verify correctness
                expected = torch.cumsum(x, dim=0)
                torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_cuda_parallel_scan_different_sizes(self):
        """Test CUDA parallel scan with different tensor sizes"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        sizes_to_test = [1, 10, 100, 1000]
        
        for size in sizes_to_test:
            with self.subTest(size=size):
                x = torch.randn(size, device="cuda", dtype=torch.float32).contiguous()
                
                # Capture stdout to verify CUDA parallel scan is used
                f = io.StringIO()
                with redirect_stdout(f):
                    result = cuda_parallel_associative_scan(
                        combine_fn=add_fn,
                        xs=x,
                        dim=0,
                        reverse=False,
                        combine_mode="pointwise"
                    )
                output = f.getvalue()
                
                # Verify the CUDA parallel implementation was used
                self.assertIn("CUDA_PARALLEL_SCAN_USED", output)
                
                # Verify correctness
                expected = torch.cumsum(x, dim=0)
                torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_cuda_parallel_scan_complex_operation(self):
        """Test CUDA parallel scan with a more complex pointwise operation"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def complex_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y * 0.5 + math.sin(y) * 0.1

        x = torch.randn(200, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=complex_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="pointwise"
            )
        output = f.getvalue()
        
        # Verify the CUDA parallel implementation was used
        self.assertIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify the result has the expected shape and is on CUDA
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, x.dtype)

    @skip_if_no_cuda
    def test_fallback_to_standard_scan_multidim(self):
        """Test that multi-dimensional tensors fall back to standard scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        # Multi-dimensional tensor should NOT use CUDA parallel scan
        x = torch.randn(10, 5, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is NOT used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=add_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="pointwise"
            )
        output = f.getvalue()
        
        # Verify the CUDA parallel implementation was NOT used
        self.assertNotIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness
        expected = torch.cumsum(x, dim=0)
        torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_fallback_to_standard_scan_reverse(self):
        """Test that reverse=True falls back to standard scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        x = torch.randn(100, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is NOT used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=add_fn,
                xs=x,
                dim=0,
                reverse=True,  # This should cause fallback
                combine_mode="pointwise"
            )
        output = f.getvalue()
        
        # Verify the CUDA parallel implementation was NOT used
        self.assertNotIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness (reverse cumsum)
        expected = torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
        torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_fallback_to_standard_scan_generic_mode(self):
        """Test that combine_mode='generic' falls back to standard scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        x = torch.randn(100, device="cuda", dtype=torch.float32).contiguous()
        
        # Capture stdout to verify CUDA parallel scan is NOT used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=add_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="generic"  # This should cause fallback
            )
        output = f.getvalue()
        
        # Verify the CUDA parallel implementation was NOT used
        self.assertNotIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness
        expected = torch.cumsum(x, dim=0)
        torch.testing.assert_close(result, expected)

    @skip_if_no_cuda
    def test_fallback_to_standard_scan_non_contiguous(self):
        """Test that non-contiguous tensors fall back to standard scan"""
        cuda_parallel_associative_scan = self._import_cuda_parallel_scan()
        
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        # Create non-contiguous tensor
        x_base = torch.randn(200, device="cuda", dtype=torch.float32)
        x = x_base[::2]  # Non-contiguous tensor
        self.assertFalse(x.is_contiguous())
        
        # Capture stdout to verify CUDA parallel scan is NOT used
        f = io.StringIO()
        with redirect_stdout(f):
            result = cuda_parallel_associative_scan(
                combine_fn=add_fn,
                xs=x,
                dim=0,
                reverse=False,
                combine_mode="pointwise"
            )
        output = f.getvalue()
        
        # Verify the CUDA parallel implementation was NOT used
        self.assertNotIn("CUDA_PARALLEL_SCAN_USED", output)
        
        # Verify correctness
        expected = torch.cumsum(x, dim=0)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    # Simple test runner
    unittest.main()