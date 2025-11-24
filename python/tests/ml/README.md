# Machine Learning Tests

Comprehensive test coverage for graph export and PyTorch utilities.

## Test Organization

Tests follow user workflow progression:

1. **TestBasicConversion** - Single graph operations
2. **TestBatchingAndDataLoaders** - Multi-graph training preparation
3. **TestAdvancedFormats** - Sparse matrix formats
4. **TestPostProcessing** - GNN output aggregation
5. **TestUtilityFunctions** - Helper functions

## Coverage Summary

**31 tests, 100% passing**

### TestBasicConversion (6 tests)
Tests fundamental graph-to-tensor conversion.

- `test_to_torch_tensors_basic` - Verifies dict keys returned
- `test_tensor_shapes_and_dtypes` - Validates shapes and data types
- `test_device_placement_cpu` - Confirms CPU placement
- `test_device_placement_cuda` - Confirms CUDA placement (requires GPU)
- `test_to_torch_geometric_basic` - PyG Data object creation
- `test_to_torch_geometric_with_label` - PyG with classification labels

**Validates:** NumPy → PyTorch conversion, device handling, PyG integration

### TestBatchingAndDataLoaders (9 tests)
Tests combining multiple graphs for training.

- `test_batch_single_graph` - Single graph batching edge case
- `test_batch_multiple_graphs` - Combining 3 graphs
- `test_batch_with_labels` - Batching with supervised labels
- `test_batch_assignment_correct` - Batch index tracking
- `test_batch_different_sizes` - Variable-size graph handling
- `test_batch_empty_raises_error` - Error on empty batch
- `test_batch_labels_mismatch_raises_error` - Label validation
- `test_batch_graphs_geometric` - PyG native batching
- `test_dataloader_creation` - DataLoader construction

**Validates:** Batch correctness, label handling, error cases, DataLoader integration

### TestAdvancedFormats (3 tests)
Tests sparse adjacency matrix formats.

- `test_sparse_coo_format` - COO (coordinate) format
- `test_sparse_csr_format` - CSR (compressed sparse row) format
- `test_sparse_format_validation` - Invalid format error handling

**Validates:** Sparse matrix creation, format conversion, validation

### TestPostProcessing (5 tests)
Tests graph-level embedding extraction.

- `test_mean_pooling` - Average aggregation
- `test_sum_pooling` - Sum aggregation
- `test_max_pooling` - Max aggregation
- `test_root_pooling` - Clause root node extraction
- `test_pooling_method_validation` - Invalid method error

**Validates:** All pooling strategies, numerical correctness, error handling

### TestUtilityFunctions (8 tests)
Tests helper and analysis functions.

- `test_node_type_masks_creation` - Mask dictionary creation
- `test_node_type_masks_are_boolean` - Boolean dtype validation
- `test_node_type_masks_select_correctly` - Mask correctness
- `test_node_type_mask_usage` - Feature filtering with masks
- `test_graph_statistics_basic` - Basic stat computation
- `test_graph_statistics_node_type_counts` - Type counting
- `test_graph_statistics_depth` - Max depth calculation
- `test_graph_statistics_usage_for_debugging` - Debugging use case

**Validates:** Node filtering, statistical analysis, debugging utilities

## Running Tests

```bash
# All ML tests
pytest python/tests/ml/test_graph_utils.py -v

# Specific test class
pytest python/tests/ml/test_graph_utils.py::TestBasicConversion -v

# Specific test
pytest python/tests/ml/test_graph_utils.py::TestBasicConversion::test_to_torch_tensors_basic -v

# With coverage
pytest python/tests/ml/test_graph_utils.py --cov=proofatlas.ml --cov-report=html
```

## Requirements

**Required:**
- PyTorch (`torch`)

**Optional (some tests skipped without):**
- PyTorch Geometric (`torch-geometric`) - 6 tests
- CUDA GPU - 1 test

## Test Results

```
31 passed, 1 warning in 2.53s
```

**Warning:** CSR sparse format is in beta (PyTorch limitation)

## What's Tested

### Conversion Correctness
- Edge indices shape: `(2, num_edges)`
- Node features shape: `(num_nodes, 20)`
- Data types: int64 (edges), float32 (features), uint8 (types)

### Batching Correctness
- Batch indices correctly track graph membership
- Node offsets properly adjusted when combining graphs
- Labels aligned with graph count

### Numerical Correctness
- Mean pooling matches manual average calculation
- Sum pooling matches manual sum
- Root pooling selects correct first node per graph

### Error Handling
- Empty batch raises `ValueError`
- Invalid format raises `ValueError`
- Label mismatch raises `ValueError`
- Invalid pooling method raises `ValueError`

## Known Limitations

1. **CSR format warning** - PyTorch sparse CSR is beta
2. **CUDA test** - Skipped if no GPU available
3. **PyG tests** - Skipped if torch-geometric not installed

## Performance

- **Execution time:** ~2.5 seconds for all 31 tests
- **No external files:** All tests use inline TPTP strings
- **Parallel safe:** Tests are independent

## Coverage Validation

Each function in `proofatlas.ml.graph_utils` has corresponding tests:

| Function | Test Count |
|----------|-----------|
| `to_torch_tensors` | 3 |
| `to_torch_geometric` | 2 |
| `to_sparse_adjacency` | 3 |
| `batch_graphs` | 7 |
| `batch_graphs_geometric` | 1 |
| `create_dataloader` | 1 |
| `extract_graph_embeddings` | 5 |
| `get_node_type_masks` | 4 |
| `compute_graph_statistics` | 4 |

**Total: 30 API tests + 1 CUDA device test = 31 tests**
