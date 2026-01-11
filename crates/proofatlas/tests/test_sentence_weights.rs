//! Integration test for loading sentence encoder weights from Python export

use std::process::Command;
use burn::module::Module;

/// Test that we can load weights exported from Python
#[test]
#[ignore = "requires Python environment with transformers installed"]
fn test_load_python_exported_weights() {
    // Export weights from Python
    let output = Command::new("python3")
        .args([
            "-c",
            r#"
from proofatlas.selectors.sentence import SentenceEncoder
import json

model = SentenceEncoder(hidden_dim=32, freeze_encoder=True)
model.export_weights('/tmp/test_sentence_weights.safetensors')
model.tokenizer.save_pretrained('/tmp/test_sentence_tokenizer')

# Save config
with open('/tmp/test_sentence_config.json', 'w') as f:
    json.dump(model.config, f)
print('Export complete')
"#,
        ])
        .output()
        .expect("Failed to run Python export");

    if !output.status.success() {
        eprintln!("Python stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Python export failed");
    }

    // Read config
    let config_str = std::fs::read_to_string("/tmp/test_sentence_config.json")
        .expect("Failed to read config");
    let config: serde_json::Value = serde_json::from_str(&config_str).expect("Failed to parse config");

    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let hidden_dim = config["encoder_dim"].as_u64().unwrap() as usize;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let num_heads = config["num_heads"].as_u64().unwrap() as usize;
    let intermediate_dim = config["intermediate_dim"].as_u64().unwrap() as usize;
    let max_position_embeddings = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let scorer_hidden_dim = config["hidden_dim"].as_u64().unwrap() as usize;

    println!("Config: vocab={}, hidden={}, layers={}, heads={}, intermediate={}, max_pos={}, scorer={}",
        vocab_size, hidden_dim, num_layers, num_heads, intermediate_dim, max_position_embeddings, scorer_hidden_dim);

    // Try to load in Burn
    use proofatlas::selectors::burn_sentence::SentenceModel;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

    let device = burn_ndarray::NdArrayDevice::Cpu;

    let model: SentenceModel<burn_ndarray::NdArray<f32>> = SentenceModel::new(
        &device,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        max_position_embeddings,
        scorer_hidden_dim,
    );

    let load_args = LoadArgs::new("/tmp/test_sentence_weights.safetensors".into())
        .with_adapter_type(AdapterType::PyTorch);

    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("Failed to load safetensors");

    let _model = model.load_record(record);
    println!("Model loaded successfully!");
}

/// Test that Python and Rust produce the same embeddings
#[test]
#[ignore = "requires Python environment with transformers installed"]
fn test_output_matches_python() {
    use burn::tensor::Tensor;
    use proofatlas::selectors::burn_sentence::SentenceModel;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

    // Export weights and compute embeddings in Python
    let output = Command::new("python3")
        .args([
            "-c",
            r#"
import torch
import json
from proofatlas.selectors.sentence import SentenceEncoder

model = SentenceEncoder(hidden_dim=32, freeze_encoder=True)
model.eval()

# Export weights
model.export_weights('/tmp/test_output_weights.safetensors')
model.tokenizer.save_pretrained('/tmp/test_output_tokenizer')

# Save config
with open('/tmp/test_output_config.json', 'w') as f:
    json.dump(model.config, f)

# Compute embeddings for test input
test_strings = ["p(X, Y)", "q(a, b) | r(c)"]
with torch.no_grad():
    embeddings = model.encode(test_strings)

# Save embeddings and input_ids for comparison
inputs = model.tokenizer(test_strings, padding=True, truncation=True, return_tensors="pt")
torch.save({
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'embeddings': embeddings,
}, '/tmp/test_output_reference.pt')

print('Reference embeddings shape:', embeddings.shape)
print('First embedding (first 5 values):', embeddings[0, :5].tolist())
"#,
        ])
        .output()
        .expect("Failed to run Python");

    println!("Python output: {}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        eprintln!("Python stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Python export failed");
    }

    // Load config
    let config_str = std::fs::read_to_string("/tmp/test_output_config.json").unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let hidden_dim = config["encoder_dim"].as_u64().unwrap() as usize;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let num_heads = config["num_heads"].as_u64().unwrap() as usize;
    let intermediate_dim = config["intermediate_dim"].as_u64().unwrap() as usize;
    let max_position_embeddings = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let scorer_hidden_dim = config["hidden_dim"].as_u64().unwrap() as usize;

    // Load model in Burn
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let model: SentenceModel<burn_ndarray::NdArray<f32>> = SentenceModel::new(
        &device,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        max_position_embeddings,
        scorer_hidden_dim,
    );

    let load_args = LoadArgs::new("/tmp/test_output_weights.safetensors".into())
        .with_adapter_type(AdapterType::PyTorch);
    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("Failed to load safetensors");
    let model = model.load_record(record);

    // Load reference data from Python
    let reference_output = Command::new("python3")
        .args([
            "-c",
            r#"
import torch
data = torch.load('/tmp/test_output_reference.pt')
print('INPUT_IDS:', data['input_ids'].tolist())
print('ATTENTION_MASK:', data['attention_mask'].tolist())
print('EMBEDDINGS:', data['embeddings'].tolist())
"#,
        ])
        .output()
        .expect("Failed to load reference");

    let ref_str = String::from_utf8_lossy(&reference_output.stdout);
    println!("Reference data:\n{}", ref_str);

    // Parse input_ids and attention_mask from Python output
    let lines: Vec<&str> = ref_str.lines().collect();
    let input_ids_line = lines.iter().find(|l| l.starts_with("INPUT_IDS:")).unwrap();
    let mask_line = lines.iter().find(|l| l.starts_with("ATTENTION_MASK:")).unwrap();
    let emb_line = lines.iter().find(|l| l.starts_with("EMBEDDINGS:")).unwrap();

    // Parse as JSON arrays
    let input_ids_json: Vec<Vec<i64>> = serde_json::from_str(
        input_ids_line.strip_prefix("INPUT_IDS: ").unwrap()
    ).unwrap();
    let mask_json: Vec<Vec<i64>> = serde_json::from_str(
        mask_line.strip_prefix("ATTENTION_MASK: ").unwrap()
    ).unwrap();
    let ref_emb_json: Vec<Vec<f64>> = serde_json::from_str(
        emb_line.strip_prefix("EMBEDDINGS: ").unwrap()
    ).unwrap();

    let batch_size = input_ids_json.len();
    let seq_len = input_ids_json[0].len();

    // Create Burn tensors
    let flat_ids: Vec<i64> = input_ids_json.into_iter().flatten().collect();
    let flat_mask: Vec<f32> = mask_json.into_iter().flatten().map(|x| x as f32).collect();

    let input_tensor: Tensor<burn_ndarray::NdArray<f32>, 2, burn::tensor::Int> = Tensor::from_data(
        burn::tensor::TensorData::new(flat_ids, [batch_size, seq_len]),
        &device,
    );
    let mask_tensor: Tensor<burn_ndarray::NdArray<f32>, 2> = Tensor::from_data(
        burn::tensor::TensorData::new(flat_mask, [batch_size, seq_len]),
        &device,
    );

    // Compute embeddings in Burn
    let burn_embeddings = model.encode(input_tensor, mask_tensor);
    let burn_emb_data: Vec<f32> = burn_embeddings.into_data().to_vec().unwrap();

    println!("Burn embeddings (first 5): {:?}", &burn_emb_data[..5]);
    println!("Python embeddings (first 5): {:?}", &ref_emb_json[0][..5]);

    // Compare (allow small tolerance for floating point differences)
    // Tolerance of 1e-3 is reasonable for cross-framework comparison
    // (differences arise from GELU approximation, accumulation order, etc.)
    let ref_flat: Vec<f64> = ref_emb_json.into_iter().flatten().collect();
    assert_eq!(burn_emb_data.len(), ref_flat.len(), "Embedding sizes don't match");

    let mut max_diff: f64 = 0.0;
    for (i, (b, r)) in burn_emb_data.iter().zip(ref_flat.iter()).enumerate() {
        let diff = (*b as f64 - *r).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-3 {
            println!("Large mismatch at index {}: Burn={}, Python={}, diff={}", i, b, r, diff);
        }
    }
    println!("Max difference: {} (tolerance: 1e-3)", max_diff);
    assert!(max_diff < 1e-3, "Embeddings differ too much: max_diff={}", max_diff);
    println!("Embeddings match within tolerance!");
}

/// List the exported weight names from Python
#[test]
#[ignore = "requires Python environment"]
fn test_list_exported_weight_names() {
    let output = Command::new("python3")
        .args([
            "-c",
            r#"
from proofatlas.selectors.sentence import SentenceEncoder
from safetensors import safe_open

model = SentenceEncoder(hidden_dim=32, freeze_encoder=True)
model.export_weights('/tmp/test_weights_list.safetensors')

with safe_open('/tmp/test_weights_list.safetensors', framework='pt') as f:
    names = sorted(f.keys())
    for name in names:
        print(name)
"#,
        ])
        .output()
        .expect("Failed to run Python");

    println!("Exported weight names:");
    println!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.status.success() {
        eprintln!("Python stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
}
