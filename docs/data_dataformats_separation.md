# Data and DataFormats Separation

## Design Principle

The `data` module and `dataformats` module are intentionally separated:

- **Data Module**: Manages collections of problems and proofs, returning raw Python objects
- **DataFormats Module**: Transforms these objects into formats suitable for machine learning models

This separation allows for:
1. Clean, reusable data management independent of model requirements
2. Multiple data format options without changing the dataset
3. Easy addition of new formats without modifying data loading

## Example Usage

### Using Data Module Alone

```python
from proofatlas.data import Problemset, DatasetConfig

# Load raw data
config = DatasetConfig.from_yaml("config.yaml")
dataset = Problemset(config, split='train')

# Get raw ProofState
state, metadata = dataset[0]
print(f"Problem: {metadata['problem_name']}")
print(f"Unprocessed clauses: {len(state.unprocessed)}")
```

### Adding DataFormat for ML Training

```python
from proofatlas.data import Problemset
from proofatlas.dataformats import get_data_format

# Load dataset
dataset = Problemset(config, split='train')

# Choose format based on model needs
graph_format = get_data_format('graph')
token_format = get_data_format('token')

# Encode the same data different ways
state, metadata = dataset[0]
graph_data = graph_format.encode_state(state)
token_data = token_format.encode_state(state)
```

### Creating a Training DataLoader

```python
from torch.utils.data import DataLoader, Dataset

class FormattedDataset(Dataset):
    """Wrapper that adds formatting to a Problemset."""
    
    def __init__(self, problemset, data_format='graph'):
        self.problemset = problemset
        self.formatter = get_data_format(data_format)
    
    def __len__(self):
        return len(self.problemset)
    
    def __getitem__(self, idx):
        state, metadata = self.problemset[idx]
        encoded = self.formatter.encode_state(state)
        return encoded, metadata

# Use in training
train_data = Problemset(config, 'train')
formatted_data = FormattedDataset(train_data, 'graph')
dataloader = DataLoader(formatted_data, batch_size=32)
```

### For Proof Training

```python
from proofatlas.data import Proofset

# Get raw proofs
proofset = Proofset("my_proofs")
proof, metadata = proofset[0]

# Format for sequence model training
formatter = get_data_format('token')
encoded_steps = []
selections = []

for i, step in enumerate(proof.steps[:-1]):
    encoded = formatter.encode_state(step.state)
    encoded_steps.append(encoded)
    selections.append(proof.steps[i+1].selected_clause or -1)
```

## Benefits

1. **Flexibility**: Switch between graph, token, or other formats without changing datasets
2. **Reusability**: Same dataset can be used with different model architectures
3. **Testing**: Easy to test data loading separately from formatting
4. **Performance**: Can cache raw data and format on-the-fly based on model needs