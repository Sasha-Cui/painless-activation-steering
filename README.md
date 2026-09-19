# Painless Activation Steering (PAS)

Activation-steering pipeline for multiple-choice benchmarks across LLMs. It builds steering vectors from model activations, sweeps layers/strengths, evaluates on standard benchmarks, and saves figures/tables for analysis.

- Paper artifacts: see `statistical_analysis.ipynb` for tables/plots used in the paper.
- Benchmarks supported include: MMLU, TruthfulQA, OpenBookQA, CommonsenseQA, ARC-Challenge, LSAT (LR/RC/AR), BBQ (all categories), ETHICS, Sycophancy, and more (see configs and `benchmark_loading_functions.py`).

## Requirements

- GPU(s). CPU execution is not supported by the current model loader.
- CUDA-compatible PyTorch (and bitsandbytes, if quantization mode is used).  This is provided in the conda env.
- Internet access for Hugging Face datasets/models.

## Installation

```bash
# create environment
conda env create -f llm-gpu-clean.yml
conda activate llm-gpu-clean
```

## Configure Hugging Face

Set `HF_HOME` directly:
```bash
export HF_HOME=/path/you/prefer
```

Authentication (create a `.env` in repo root):
```bash
# .env
hf_token=hf_xxx_your_token_here
```

## Quickstart

Run an experiment with a provided config/Test the environment:
```bash
python main.py -c configs/scale_up_configs/test.yaml
```

Override selected YAML fields via CLI:
```bash
python main.py -c configs/scale_up_configs/scale_up_llama8b.yaml \
  --seed 42 \
  --max_sample_size 1000 \
  --n_train_proportion 0.6 --n_val_proportion 0.2 --n_test_proportion 0.2
```

Tip: For GPU-constrained environments, prefer 4-bit or 8-bit quantization in your YAML.  Start with a smaller model like `TinyLlama-1.1B-Chat-v1.0` for testing runs.

## Configuration

A YAML example:
```yaml
seed: 39
device: cuda             # cuda | cpu (loader requires GPU)
output_dir: output_scale_up

quantization: 4bit       # 4bit | 8bit | none
model_name: meta-llama/Llama-3.1-8B-Instruct

max_sample_size: 4000
n_train_proportion: 0.6
n_val_proportion:   0.2
n_test_proportion:  0.2

benchmarks:
  - TruthfulQA
  - CommonsenseQA

layers_to_try:  [8, 11, 13, 14, 15, 16, 17, 19, 22, 25]
steer_strengths: [0.25, 0.5, 1, 4, 7, 10, 13, 16, 19, 22, 25, 32]
methods:
  - BAS_full_mcq
  - iBAS_all
  - iBAS_wrong_only

steer_target: residual    # residual | mlp | self_attn | post_attention_layernorm

# Optional modes
icl_k: 0                  # >0 enables ICL baseline/eval using wrong-train exemplars
catastrophic_test: false
# target_benchmark: ...   # required if catastrophic_test: true
```

- **Benchmarks**: must appear in the implemented `benchmark_map` (see `benchmark_loading_functions.py`), including `MMLU`, `TruthfulQA`, `OpenBookQA`, `CommonsenseQA`, `ARCChallenge`, `LSAT`, `BBQ` categories (`Age`, `SES`, etc.), `ETHICS` subsets, `Sycophancy`, and many more.
- **Methods**: supported `BAS_full_mcq`, `iBAS_all`, and `iBAS_wrong_only', which correspond to 'PASf', 'iPASa', and 'iPASwo' in the paper, respectively.
- **Steering target**: one of `residual`, `mlp`, `self_attn`, `post_attention_layernorm` (alias `post_attn` is accepted internally).
- **ICL mode**: set `icl_k > 0` to evaluate with ICL (k in-context exemplars drawn from wrong-train items).
- **Catastrophic forgetting test**: set `catastrophic_test: true` and specify `target_benchmark`.

## Outputs

Results are written under:
```
<output_dir>/<model_name_sanitized>/<steer_target or subdir>/seed_<seed>/
```
Key folders/files:
- `logfile_*.txt`: run log with environment info.
- `steer_vectors/<benchmark>/...pt`: saved vectors per method/layer/strength.
- `predictions/<benchmark>/<method>/predictions.csv`: item-level predictions (+ ICL meta if enabled).
- `figures/`: per-benchmark heatmaps and test-set bar plots.
- `tables/`: per-benchmark tables, plus cross-benchmark:
  - `cross_benchmark_summary.csv`
  - `cross_benchmark_flips.csv`

## Reproducing paper figures/tables

### Running experiments

- Table 3 and 5:
```bash
for seed in {1..15}; do
  python main.py -c configs/scale_up_configs/scale_up_<MODEL_NAME>.yaml --seed $seed
done
```
Replace `<MODEL_NAME>` with each of the three models (e.g., `DeepSeek-R1-Distill-Llama-8B`, `Llama-3.1-8B-Instruct`, or `Nous-Hermes-2-Mistral-7B-DPO`).
Note that Table 3 and Table 5 use different sets of benchmarks.

- Table 4:
```bash
for seed in {1..15}; do
  python main.py -c configs/scale_up_catastrophic/scale_up_<MODEL_NAME>.yaml --seed $seed
done
```
Replace `<MODEL_NAME>` with each of the three models (e.g., `DeepSeek-R1-Distill-Llama-8B`, `Llama-3.1-8B-Instruct`, or `Nous-Hermes-2-Mistral-7B-DPO`).

- Table 6-8:
```bash
for seed in {1..15}; do
  python main.py -c configs/scale_up_catastrophic/scale_up_<MODEL_NAME>.yaml --seed $seed --steer_target <steer_target>
done
```
Replace `<MODEL_NAME>` with each of the three models (e.g., `DeepSeek-R1-Distill-Llama-8B`, `Llama-3.1-8B-Instruct`, or `Nous-Hermes-2-Mistral-7B-DPO`).
Replace '<steer_target>' with each of the three steering targets (e.g., `self attn`, `post attn`, or `mlp`).

- Table 9-10:
```bash
for seed in {1..15}; do
  python main.py -c configs/scale_up_ICL/icl_scale_up_<MODEL_NAME>.yaml --seed $seed
done
```
Replace `<MODEL_NAME>` with each of the three models (e.g., `DeepSeek-R1-Distill-Llama-8B`, `Llama-3.1-8B-Instruct`, or `Nous-Hermes-2-Mistral-7B-DPO`).

- Comparison with SFT (Supervised Fine Tuning):
```
Replace `<MODEL_NAME>` with `main_vicuna-7b.yaml` and `main_vicuna-7b-fine-tuning_truthfulQA_128_20`
```

- Run Duration Analysis:
Use `configs/timer/scale_up_<MODEL_NAME>.yaml` configurations and submit jobs as above.  Use `timer_analysis.ipynb` to collect and compile the run duration data.

- Sample Size Sensitivity Analysis:
Use `configs/sample_size_sensitivity/scale_up_<MODEL_NAME>.yaml` for config files and use

```bash
splits=(
    "12 4 800"
    "24 8 800"
    "48 12 800"
    "75 25 800"
    "150 50 800"
    "300 100 800"
    "600 200 800"
    "1200 400 800"
    "2400 800 800"
)

for split in "${splits[@]}"; do
    read -r ntrain nval ntest <<<"$split"
    total=$((ntrain+nval+ntest))
    train_prop=$(python -c "print(round($ntrain/$total,6))")
    val_prop=$(python -c "print(round($nval/$total,6))")
    test_prop=$(python -c "print(round($ntest/$total,6))")

    echo ">>> Running $model_yaml seed=$seed with split $ntrain/$nval/$ntest"
    python main.py -c "$model_yaml" --seed "$seed" \
      --max_sample_size="$total" \
      --n_train_proportion="$train_prop" \
      --n_val_proportion="$train_prop" \
      --n_test_proportion="$test_prop"
done
```
to run the experiments with different sample sizes.


### Creating tables and plots

Open `statistical_analysis.ipynb` to regenerate tables and plots used in the paper.

## Troubleshooting

- **bitsandbytes/CUDA errors**: ensure the provided conda env is active and matches your GPU/CUDA. Multi-GPU is auto-handled via `device_map="auto"`.
- **CPU run**: the loader currently requires a CUDA device.
- **Hugging Face auth**: ensure `.env` contains `hf_token` or log in via `huggingface-cli login`.
- **Cache/storage**: set `HF_HOME` to a disk with sufficient space.


# Main Reference

If you find Painless Activation Steering (PAS) useful for your work, please consider citing our [paper](https://www.arxiv.org/abs/2509.22739).
