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
OPENAI_API_KEY=sk-xxx_your_key_here   # only needed for the open-ended experiment (Table 11)
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

- Tables 2 and 3, and Figure 2 (CAA baseline):
```bash
# 1. generate the contrastive prompt pairs with each model (10 per task);
#    skip this step to reuse the pairs shipped in prompt_pairs_outputs/
python generate_caa_pairs.py --model <HF_MODEL> \
    --generators contrastive_prompt_generators.csv \
    --out prompt_pairs_outputs/<MODEL_STEM>_contrastive_pairs_expanded.csv
python generate_caa_pairs.py --model <HF_MODEL> \
    --generators contrastive_prompt_generators_intelligence.csv \
    --out prompt_pairs_outputs/<MODEL_STEM>_contrastive_pairs_expanded.csv

# 2. run the baseline (one steering vector per task, seed 39)
python main.py -c configs/CAA/CAA_per_task_<MODEL>.yaml
python main.py -c configs/CAA/CAA_per_task_intelligence_<MODEL>.yaml
```
`<MODEL>` is `Nous`, `DeepSeek` or `Llama`. The baseline builds one vector per task from that task's pairs, as in CAA (Rimsky et al., 2024).

- Table 3 (positive/negative ratios of the PAS training sets):
```bash
python compute_pos_neg_ratios.py                 # all 3 models x 15 tasks x 15 seeds
python compute_pos_neg_ratios.py --seeds 1       # one seed per model (much faster)
```
Writes `pos_neg_ratios.csv`; the version behind Table 3 is included as `results/pos_neg_ratios/pos_neg_ratios.csv`.

- Table 13 (open-ended generation):
```bash
python main_open_ended.py -c configs/open_ended/open_ended_Nous-Hermes-2-Mistral-7B-DPO.yaml
python main_open_ended.py -c configs/open_ended/open_ended_DeepSeek-R1-Distill-Llama-8B.yaml
```
Each config contains the seed, sample size, and layer/strength grid used for Table 11. Steering vectors and hyperparameters are selected on the multiple-choice train/validation splits; the test split is converted to open-ended prompts, the model answers with greedy decoding, and GPT-4o grades each answer 0/1 against the reference answer. This requires `OPENAI_API_KEY` (see above) and makes paid OpenAI API calls.

Results are written to `output_scale_up/<model>/open_ended/residual/seed_<seed>/tables/open_ended_test_scores.csv`. In Table 13, `Raw` is the `raw_model` column, and `iPASa` / `iPASwo` are the `iBAS_all` / `iBAS_wrong_only` columns minus `raw_model`. GPT-4o grading is not fully deterministic, so reruns may differ slightly from the reported values.

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

- Figure 2 (PAS vs. CAA on behavior tasks) and Tables 2-3: the per-seed summary tables are included in `results/figure2/` (PAS: `output_scale_up/<model>/residual/seed_{1..15}`; CAA: `output_CAA_per_task/<model>/residual/seed_39`), and the positive/negative counts in `results/pos_neg_ratios/`. Run the first section of the notebook to produce `plots/figure2.pdf`, `tables/table_main_summary.tex` (Table 2) and `tables/table_pos_neg_ratio.tex` (Table 3).

## Troubleshooting

- **bitsandbytes/CUDA errors**: ensure the provided conda env is active and matches your GPU/CUDA. Multi-GPU is auto-handled via `device_map="auto"`.
- **CPU run**: the loader currently requires a CUDA device.
- **Hugging Face auth**: ensure `.env` contains `hf_token` or log in via `huggingface-cli login`.
- **Cache/storage**: set `HF_HOME` to a disk with sufficient space.


# Main Reference

If you find Painless Activation Steering (PAS) useful for your work, please consider citing our [paper](https://www.arxiv.org/abs/2509.22739).
