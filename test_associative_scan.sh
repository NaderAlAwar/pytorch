#!/bin/bash

# Script to run ONLY associative_scan related tests using precise pytest commands
# This avoids running unrelated tests in the same files

set -e  # Exit on any error

echo "======================================================================"
echo "Running ONLY associative_scan related tests with pytest"
echo "======================================================================"

cd "$(dirname "$0")"

echo ""
echo "1. Running functorch/test_control_flow.py associative_scan tests..."
echo "   - AssociativeScanTests class (34 tests)"
echo "   - TestControlFlow::test_scan_associative_scan"
python -m pytest test/functorch/test_control_flow.py::AssociativeScanTests --tb=short
python -m pytest test/functorch/test_control_flow.py::TestControlFlow::test_scan_associative_scan --tb=short

echo ""
echo "2. Running inductor/test_torchinductor.py custom scan tests..."
echo "   - 4 custom scan tests from TestCase class"
python -m pytest test/inductor/test_torchinductor.py -k "test_custom_scan" --tb=short

echo ""
echo "3. Running inductor/test_control_flow.py associative_scan tests..."
echo "   - AssociativeScanTests class (1 test)"
python -m pytest test/inductor/test_control_flow.py::AssociativeScanTests --tb=short

echo ""
echo "4. Running inductor/test_op_dtype_prop.py assoc_scan test..."
echo "   - TestCase::test_assoc_scan"
python -m pytest test/inductor/test_op_dtype_prop.py::TestCase::test_assoc_scan --tb=short

echo ""
echo "5. Running inductor/test_cuda_repro.py non_commutative_scan_op test..."
echo "   - CudaReproTests::test_non_commutative_scan_op"
python -m pytest test/inductor/test_cuda_repro.py::CudaReproTests::test_non_commutative_scan_op --tb=short

echo ""
echo "6. Running export/test_export.py export associative_scan tests..."
echo "   - 3 export associative_scan tests from TestExport class"
python -m pytest test/export/test_export.py -k "test_export_associative_scan" --tb=short

echo ""
echo "7. Running dynamo/test_misc.py hash_hop test..."
echo "   - test_hash_hop (uses associative_scan)"
python -m pytest test/dynamo/test_misc.py -k "test_hash_hop" --tb=short

echo ""
echo "======================================================================"
echo "All associative_scan tests completed!"
echo "======================================================================"
echo "Summary of tests run:"
echo "  - functorch: 35 tests (34 AssociativeScanTests + 1 test_scan_associative_scan)"
echo "  - inductor torchinductor: 4 custom scan tests"
echo "  - inductor control_flow: 1 associative_scan test"
echo "  - inductor op_dtype_prop: 1 assoc_scan test"
echo "  - inductor cuda_repro: 1 non_commutative_scan_op test"
echo "  - export: 3 export associative_scan tests"
echo "  - dynamo: 1 hash_hop test"
echo "  Total: ~46 associative_scan related tests"