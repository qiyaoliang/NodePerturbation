# Node Perturbation vs Weight Perturbation on GSM8K

This repository contains implementations of two evolutionary strategies approaches for optimizing language models on the GSM8K mathematical reasoning dataset:

- **Node Perturbation (NP)**: Perturbs activations during forward passes
- **Weight Perturbation (WP)**: Perturbs model weights directly

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU(s) recommended
- At least 16GB RAM per GPU for 1.5B parameter models

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NodePerturbation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- **torch**: PyTorch for deep learning
- **transformers**: Hugging Face transformers library
- **accelerate**: For distributed training
- **datasets**: Hugging Face datasets library (for GSM8K)
- **numpy**: Numerical computing
- **tqdm**: Progress bars
- **pandas**: Data analysis (optional)
- **sentencepiece**: Tokenization support
- **protobuf**: Protocol buffers

## Dataset

Both scripts automatically download the GSM8K dataset from Hugging Face on first run. No manual download is required.

## Usage

### Weight Perturbation (WP)

Basic single-GPU run:

```bash
python wp_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --n_train 64 --n_eval 100 \
  --n_iters 100 --pop_size 30 \
  --sigma 0.001 --alpha 0.001 \
  --visualization_dir ./out_wp_gsm8k
```

Multi-GPU distributed run (recommended):

```bash
accelerate launch --num_processes=8 wp_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --hf_cache_dir /path/to/cache \
  --mixed_precision bf16 \
  --n_train 64 --n_eval 100 \
  --n_iters 100 --pop_size 30 \
  --sigma 0.001 --alpha 0.001 \
  --meta_seed 42 \
  --gpu_threads 1 \
  --eval_interval 20 \
  --visualization_dir ./out_wp_gsm8k
```

### Node Perturbation (NP)

Basic single-GPU run:

```bash
python np_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --n_train 64 --n_eval 100 \
  --n_iters 100 --pop_size 30 \
  --sigma 0.005 --alpha 0.001 \
  --np_include mlp --last_k 1 \
  --visualization_dir ./out_np_gsm8k
```

Multi-GPU distributed run (recommended):

```bash
accelerate launch --num_processes=8 np_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --hf_cache_dir /path/to/cache \
  --mixed_precision bf16 \
  --n_train 64 --n_eval 100 \
  --n_iters 100 --pop_size 30 \
  --sigma 0.005 --alpha 0.001 \
  --np_include mlp --last_k 1 \
  --np_fast_noise True \
  --np_sync_every 1 \
  --visualization_dir ./out_np_gsm8k
```

## Command-Line Arguments

### Common Arguments (Both Scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model name |
| `--hf_cache_dir` | `/opt/dlami/nvme/catherine` | Cache directory for models |
| `--mixed_precision` | `fp16` | Precision mode: `no`, `fp16`, or `bf16` |
| `--gpu_threads` | `1` | Parallel threads per GPU |
| `--n_train` | `64` | Number of training samples |
| `--n_eval` | `100` | Number of evaluation samples |
| `--n_iters` | `100` | Number of ES iterations |
| `--pop_size` | `30` | Population size for ES |
| `--sigma` | `0.001` (WP) / `0.005` (NP) | Perturbation noise standard deviation |
| `--alpha` | `0.001` | Learning rate / step size |
| `--meta_seed` | `42` | Random seed for reproducibility |
| `--eval_interval` | `20` | Evaluate on test set every N iterations |
| `--skip_eval` | `False` | Skip periodic evaluation |
| `--verbose` | `False` | Enable verbose logging |
| `--aggressive_gc` | `False` | Enable aggressive garbage collection |
| `--visualization_dir` | `./out_*_gsm8k` | Output directory |

### NP-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--np_include` | `all` | Which layers to perturb: `head`, `attn`, `mlp`, or `all` |
| `--last_k` | `0` | Restrict to last K transformer blocks (0 = all) |
| `--np_fast_noise` | `True` | Use vectorized noise generation (faster, deterministic) |
| `--np_sync_every` | `1` | All-reduce gradients every K iterations |
| `--np_micro_batch_size` | `4` | Micro-batch size for replay (legacy, unused in cached replay) |
| `--np_max_tok_chunk` | `256` | Token chunk size (legacy, unused in cached replay) |

## Output

Both scripts produce:

1. **Reward History CSV**: `{visualization_dir}/reward_history.csv`
   - Columns: `iteration`, `candidate_index`, `seed`, `train_reward`
   - Tracks all candidate rewards across iterations

2. **Final Model**: `{visualization_dir}/final_model/`
   - Saved model checkpoint after training
   - Compatible with Hugging Face transformers

3. **Baseline Completions**: `/tmp/wp_baseline_completions.txt`
   - Sample completions before optimization
   - Useful for debugging and comparison

## Memory Management

For large models or limited GPU memory:

- Reduce `--gpu_threads` to 1
- Use `--mixed_precision bf16` (recommended for modern GPUs)
- Reduce `--pop_size` (e.g., 10-20 instead of 30)
- Enable `--aggressive_gc`
- Reduce `--n_train` samples

## Multi-GPU Setup

For distributed training across multiple GPUs:

1. **Hugging Face Accelerate** (recommended):
```bash
accelerate launch --num_processes=<num_gpus> <script>.py <args>
```

2. The scripts automatically handle:
   - Model sharding across GPUs
   - Gradient synchronization
   - Seed broadcasting
   - Reward aggregation

## Differences Between NP and WP

| Aspect | Node Perturbation (NP) | Weight Perturbation (WP) |
|--------|------------------------|--------------------------|
| **What's perturbed** | Activations during forward pass | Model weights |
| **Update method** | Gradient-based replay | Direct weight update |
| **Memory** | Higher (stores generated IDs) | Lower |
| **Speed** | Slower (requires replay) | Faster |
| **Selectivity** | Can target specific layers (`--np_include`) | Perturbs all parameters |
| **Sigma range** | Typically higher (0.005-0.01) | Typically lower (0.001) |

## Troubleshooting

### Out of Memory Errors
- Reduce `--pop_size`
- Set `--gpu_threads 1`
- Use `--mixed_precision bf16`
- Reduce `--n_train`

### Slow Training
- Increase `--gpu_threads` (if memory allows)
- Use multiple GPUs with `accelerate launch`
- For NP: verify `--np_fast_noise True`

### Reproducibility Issues
- Set `--meta_seed` to a fixed value
- Use deterministic mode (automatically enabled)
- Ensure same CUDA version across runs

### Dataset Download Fails
- Check internet connection
- Set `HF_DATASETS_OFFLINE=0` environment variable
- Manually download GSM8K and specify local path