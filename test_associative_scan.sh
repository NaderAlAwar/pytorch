#!/bin/bash

# Usage: ./test_associative_scan.sh. Set env variable `PYTORCH_ASSOCIATIVE_SCAN_CUDA_PARALLEL=1` to use the
# CUDA parallel implementation of associative_scan.

# On my machine I also had to set `CUDA_HOME=/usr/local/cuda-12.8` or the scan doesn't build.

# Script to run ONLY associative_scan related tests using precise pytest commands
# This avoids running unrelated tests in the same files

set -e  # Exit on any error

# Create a temporary file to capture all output
TEMP_LOG=$(mktemp)
trap "rm -f $TEMP_LOG" EXIT

echo "======================================================================"
echo "Running ONLY associative_scan related tests with pytest"
echo "======================================================================"

cd "$(dirname "$0")"

echo ""
echo "1. Running functorch/test_control_flow.py associative_scan tests..."
echo "   - AssociativeScanTests class (34 tests)"
echo "   - TestControlFlow::test_scan_associative_scan"
python -m pytest -s test/functorch/test_control_flow.py::AssociativeScanTests --tb=short 2>&1 | tee -a "$TEMP_LOG"
python -m pytest -s test/functorch/test_control_flow.py::TestControlFlow::test_scan_associative_scan --tb=short 2>&1 | tee -a "$TEMP_LOG"

export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 # needed for the torchinductor tests

echo ""
echo "2. Running inductor/test_torchinductor.py custom scan tests..."
echo "   - 4 custom scan tests from TestCase class"
python -m pytest -s test/inductor/test_torchinductor.py -k "test_custom_scan" --tb=short 2>&1 | tee -a "$TEMP_LOG"

unset LD_PRELOAD

echo ""
echo "3. Running inductor/test_control_flow.py associative_scan tests..."
echo "   - AssociativeScanTests class (1 test)"
python -m pytest -s test/inductor/test_control_flow.py::AssociativeScanTests --tb=short 2>&1 | tee -a "$TEMP_LOG"

echo ""
echo "4. Running inductor/test_op_dtype_prop.py assoc_scan test..."
echo "   - TestCaseCUDA::test_assoc_scan_cuda"
python -m pytest -s test/inductor/test_op_dtype_prop.py::TestCaseCUDA::test_assoc_scan_cuda --tb=short 2>&1 | tee -a "$TEMP_LOG"

echo ""
echo "5. Running inductor/test_cuda_repro.py non_commutative_scan_op test..."
echo "   - CudaReproTests::test_non_commutative_scan_op"
python -m pytest -s test/inductor/test_cuda_repro.py::CudaReproTests::test_non_commutative_scan_op --tb=short 2>&1 | tee -a "$TEMP_LOG"

echo ""
echo "6. Running export/test_export.py export associative_scan tests..."
echo "   - 3 export associative_scan tests from TestExport class"
python -m pytest -s test/export/test_export.py -k "test_export_associative_scan" --tb=short 2>&1 | tee -a "$TEMP_LOG"

export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 # needed for the dynamo tests

echo ""
echo "7. Running dynamo/test_misc.py hash_hop test..."
echo "   - test_hash_hop (uses associative_scan)"
python -m pytest -s test/dynamo/test_misc.py -k "test_hash_hop" --tb=short 2>&1 | tee -a "$TEMP_LOG"

unset LD_PRELOAD

echo ""
echo "8. Running higher_order_ops/test_cuda_parallel_associative_scan.py tests..."
echo "   - TestCudaParallelAssociativeScan class (11 tests specifically for CUDA parallel scan)"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 # needed for CUDA CCCL parallel
python -m pytest -s test/higher_order_ops/test_cuda_parallel_associative_scan.py --tb=short 2>&1 | tee -a "$TEMP_LOG"
unset LD_PRELOAD

echo ""
echo "======================================================================"
echo "All associative_scan tests completed!"
echo "======================================================================"

# Count CUDA parallel scan usage
CUDA_PARALLEL_COUNT=$(grep -c "CUDA_PARALLEL_SCAN_USED:" "$TEMP_LOG" 2>/dev/null || echo "0")
# Ensure it's a valid integer
CUDA_PARALLEL_COUNT=$(echo "$CUDA_PARALLEL_COUNT" | tr -d '\n\r' | head -1)

echo ""
echo "SUMMARY:"
echo "========"
if [ "$CUDA_PARALLEL_COUNT" -gt 0 ]; then
    echo "✅ CUDA Parallel Scan was used: $CUDA_PARALLEL_COUNT times"
    echo ""
    echo "Details of CUDA parallel usage:"
    grep "CUDA_PARALLEL_SCAN_USED:" "$TEMP_LOG" | sed 's/^/  /'
else
    echo "❌ CUDA Parallel Scan was NOT used (all tests used fallback implementation)"
    echo ""
    echo "The CUDA parallel scan currently requires the following conditions to be met:"
    echo "  1. Set environment variable: PYTORCH_ASSOCIATIVE_SCAN_CUDA_PARALLEL=1"
    echo "  2. Tests use 1D contiguous CUDA tensors"
    echo "  3. combine_mode='pointwise'"
    echo "  4. reverse=False"
fi

echo ""
echo "Total associative_scan calls that could potentially use CUDA parallel:"
echo "  (This includes both CUDA parallel and fallback implementations)"

# Count all associative scan related test executions by parsing pytest summary lines
TOTAL_TESTS_RUN=$(grep -o "[0-9]\+ passed" "$TEMP_LOG" 2>/dev/null | awk '{sum += $1} END {print sum+0}')
TOTAL_TESTS_RUN=$(echo "$TOTAL_TESTS_RUN" | tr -d '\n\r' | head -1)
# Ensure it's a valid number, default to 0 if empty
if [ -z "$TOTAL_TESTS_RUN" ] || [ "$TOTAL_TESTS_RUN" = "" ]; then
    TOTAL_TESTS_RUN="0"
fi
echo "  Tests executed: $TOTAL_TESTS_RUN"

echo ""
echo "======================================================================"