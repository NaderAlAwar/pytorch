# mypy: allow-untyped-defs
from typing import Callable

import cuda.cccl.parallel.experimental as parallel

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.associative_scan import associative_scan, _fake_associative_scan


def initialize_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    d_input: torch.Tensor,
    dim: int,
    reverse: bool,
) -> torch.Tensor:
    """
    This function builds cuda.cccl.parallel scan and allocates temp storage,
    d_output, and h_init for it.
    """

    h_init = torch.zeros(1, dtype=d_input.dtype).numpy()
    d_output = torch.empty_like(d_input)

    if reverse:
        d_input = parallel.iterators.ReverseInputIterator(d_input)
        d_output = parallel.iterators.ReverseOutputIterator(d_output)

    scanner = parallel.make_inclusive_scan(d_output, d_output, combine_fn, h_init)
    temp_storage_size = scanner(None, d_input, d_output, d_input.size(dim), h_init)
    d_temp_storage = torch.empty(temp_storage_size, dtype=torch.uint8).cuda()
    scanner(d_temp_storage, d_input, d_output, d_input.size(dim), h_init)

    return scanner, d_temp_storage, d_output, h_init


def associative_scan_impl(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    d_input: torch.Tensor,
    dim: int,
    reverse: bool,
) -> torch.Tensor:

    scanner, d_temp_storage, d_output, h_init = initialize_scan(combine_fn, d_input, dim, reverse)

    if reverse:
        h_init[0] = d_input[-1]
        current_input_it = parallel.iterators.ReverseInputIterator(d_input[:-1])
        current_output_it = parallel.iterators.ReverseOutputIterator(d_output[:-1])
    else:
        h_init[0] = d_input[0]
        current_input_it = d_input[1:]
        current_output_it = d_output[1:]

    scanner(d_temp_storage, current_input_it, current_output_it, d_input.size(dim) - 1, h_init)

    if reverse:
        d_output.data[-1] = h_init.item()
    else:
        d_output.data[0] = h_init.item()

    return d_output


def cuda_parallel_associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    A CUDA-optimized parallel implementation of associative scan that preserves
    the same semantics as the standard associative_scan.

    This implementation currently falls back to the standard associative_scan
    to ensure semantic compatibility. Future versions will implement a
    specialized CUDA parallel algorithm while maintaining the same behavior.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.

    Returns:
        torch.Tensor: The result of the associative scan operation.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = cuda_parallel_associative_scan(add, x, dim)

    .. note::
        This implementation currently falls back to the standard associative_scan
        implementation to ensure semantic compatibility during development.
    """

    if combine_mode == "pointwise" and isinstance(xs, torch.Tensor) and xs.device.type == "cuda" and xs.is_contiguous() and xs.ndim == 1:
        # print("CUDA_PARALLEL_SCAN_USED: Using CUDA parallel associative scan")
        return associative_scan_impl(combine_fn, xs, dim, reverse)

    # For now, fall back to the standard associative_scan implementation
    # This ensures we preserve the exact same semantics and behavior
    return associative_scan(
        combine_fn=combine_fn,
        xs=xs,
        dim=dim,
        reverse=reverse,
        combine_mode=combine_mode,
    )


# Re-export _fake_associative_scan for compatibility with tests that import it directly
__all__ = ["cuda_parallel_associative_scan", "_fake_associative_scan"]
