#!/usr/bin/env python
"""
compute_pos_neg_ratios.py
-------------------------
Exact positive/negative sample counts used to build PAS steering vectors.

For each (model, benchmark, seed) this rebuilds the train split exactly as the
experiments do, runs the unsteered model on it, and counts correct vs. incorrect
answers. Those counts ARE the positive/negative set sizes:

  PASf   (BAS_full_mcq)      positives = correct train items, negatives = incorrect
  iPASa  (iBAS_all)          same counts as PASf (only the prompt text differs)
  iPASwo (iBAS_wrong_only)   positives = negatives = incorrect train items  -> always 1:1

See `build_bas_full_mcq`, `build_ibas_all` and `build_ibas_wrong_only` in
build_steer_vectors_functions.py.

Defaults reproduce the runs behind Figure 2 and the main tables (max_sample_size
4000, 0.6/0.2/0.2 split, no quantization, residual steer target, seeds 1-15).
Decoding is greedy and splits are seed-determined, so the counts are exactly the
ones the original runs used.

Requires a GPU and `hf_token` in .env.

Examples
--------
# everything behind Figure 2 (3 models x 15 benchmarks x 15 seeds)
python compute_pos_neg_ratios.py

# one seed per model (enough to get the ratios to ~1 pp), smaller job
python compute_pos_neg_ratios.py --seeds 1

# a single model, selected benchmarks, custom output file
python compute_pos_neg_ratios.py \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks TruthfulQA Sycophancy \
    --out pos_neg_ratios_llama.csv

Output: one CSV row per (model, seed, benchmark) with n_train, n_pos, n_neg,
pos_neg_ratio and train_acc, plus val_acc for comparison with the summary tables.
The run is resumable: existing rows in the output CSV are skipped.
"""
from machine_environments import *  # Machine Environment Settings; resets cache directory
from preambles import *             # common imports (pandas, torch, tqdm, Path, os, sys, ...)
from benchmark_loading_functions import *   # load_benchmark, benchmark_map
from load_models_and_data import *          # load_lm
from model_prediction import predict_mcq_letters

import argparse
import csv as _csv
import gc
import time as _time
from dotenv import load_dotenv

# The three models and 15 behavior tasks reported in Figure 2.
DEFAULT_MODELS = [
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
DEFAULT_BENCHMARKS = [
    "DisabilityStatus", "GenderIdentity", "Nationality", "PhysicalAppearance",
    "RaceEthnicity", "RaceXGender", "RaceXSES", "Religion", "SES", "SexualOrientation",
    "ETHICS_Commonsense", "ETHICS_Deontology", "ETHICS_Justice",
    "Sycophancy", "TruthfulQA",
]
FIELDS = [
    "model", "seed", "benchmark", "n_train", "n_pos", "n_neg",
    "pos_neg_ratio", "train_acc", "n_val", "val_acc",
    "pasf_ipasa_pos_neg_ratio", "ipaswo_pos_neg_ratio",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face model ids")
    ap.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, help="names from benchmark_map")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 16)), help="seeds to evaluate")
    ap.add_argument("--max_sample_size", type=int, default=4000)
    ap.add_argument("--n_train_proportion", type=float, default=0.6)
    ap.add_argument("--n_val_proportion", type=float, default=0.2)
    ap.add_argument("--n_test_proportion", type=float, default=0.2)
    ap.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="pos_neg_ratios.csv", help="output CSV (appended; existing rows skipped)")
    ap.add_argument("--skip_val", action="store_true", help="skip the validation-split pass (faster)")
    ap.add_argument("--summary_out", default=None,
                    help="optional CSV for the per-(model, benchmark) summary table")
    return ap.parse_args()


def load_done(path):
    """Rows already computed, so the script can be resumed after an interruption."""
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for row in _csv.DictReader(fh):
                done.add((row["model"], int(row["seed"]), row["benchmark"]))
    return done


def append_row(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def count_split(df, model, tok, device):
    """Return (n, n_correct) for one split under greedy unsteered decoding."""
    preds = pd.Series(predict_mcq_letters(df, model, tok, device=device), index=df.index)
    n_correct = int((preds == df["correct_letter"]).sum())
    return len(df), n_correct


def main():
    args = parse_args()
    load_dotenv()
    token = os.getenv("hf_token")
    if token:
        login(token=token)

    unknown = [b for b in args.benchmarks if b not in benchmark_map]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}")

    print(f">>> job started {_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f">>> models={args.models}", flush=True)
    print(f">>> seeds={args.seeds} benchmarks={len(args.benchmarks)} out={args.out}", flush=True)

    done = load_done(args.out)
    if done:
        print(f">>> {len(done)} rows already in {args.out}; those are skipped.", flush=True)

    for model_name in args.models:
        todo = [(s, b) for s in args.seeds for b in args.benchmarks
                if (model_name, s, b) not in done]
        if not todo:
            print(f">>> {model_name}: nothing to do.", flush=True)
            continue

        print(f"\n>>> Loading {model_name} ({len(todo)} model/seed/benchmark combinations to go)", flush=True)
        model, tok = load_lm(name=model_name, device=args.device, quantization=args.quantization)

        try:
            for i, (seed, bench) in enumerate(todo, start=1):
                t0 = _time.time()
                # Same splits as the experiments: seed-determined shuffle, identical proportions.
                train_df, val_df, _test_df, _meta = load_benchmark(
                    bench,
                    max_sample_size=args.max_sample_size,
                    train_prop=args.n_train_proportion,
                    val_prop=args.n_val_proportion,
                    test_prop=args.n_test_proportion,
                    seed=seed,
                )

                n_train, n_pos = count_split(train_df, model, tok, args.device)
                n_neg = n_train - n_pos

                if args.skip_val:
                    n_val, val_acc = "", ""
                else:
                    n_val, n_val_correct = count_split(val_df, model, tok, args.device)
                    val_acc = round(n_val_correct / n_val, 6) if n_val else ""

                row = {
                    "model": model_name,
                    "seed": seed,
                    "benchmark": bench,
                    "n_train": n_train,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    # PASf and iPASa both use (correct : incorrect); iPASwo is 1:1 by construction.
                    "pos_neg_ratio": round(n_pos / n_neg, 6) if n_neg else "inf",
                    "train_acc": round(n_pos / n_train, 6) if n_train else "",
                    "n_val": n_val,
                    "val_acc": val_acc,
                    "pasf_ipasa_pos_neg_ratio": round(n_pos / n_neg, 6) if n_neg else "inf",
                    "ipaswo_pos_neg_ratio": 1.0,
                }
                append_row(args.out, row)
                print(f"    [{i}/{len(todo)}] {bench:20s} seed {seed:>2}: n_train={n_train:5d} "
                      f"pos={n_pos:5d} neg={n_neg:5d} ratio={row['pos_neg_ratio']} "
                      f"train_acc={row['train_acc']} ({_time.time() - t0:.0f}s)", flush=True)
        finally:
            # Free the GPU before loading the next model.
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summarise(args.out, args.summary_out)
    print(f"\n>>> job finished {_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


def summarise(path, summary_out=None):
    """Print mean ratio per model and per (model, benchmark)."""
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df["pos_neg_ratio"] = pd.to_numeric(df["pos_neg_ratio"], errors="coerce")

    print("\n================ positives : negatives (PASf and iPASa) ================")
    per_task = (df.groupby(["model", "benchmark"])
                  .agg(seeds=("seed", "nunique"), n_train=("n_train", "mean"),
                       pos=("n_pos", "mean"), neg=("n_neg", "mean"),
                       train_acc=("train_acc", "mean"), ratio=("pos_neg_ratio", "mean")))
    with pd.option_context("display.width", 0, "display.max_rows", None):
        print(per_task.round(3))
    if summary_out:
        per_task.round(6).to_csv(summary_out)
        print(f"\nPer-(model, benchmark) summary written to {summary_out}")

    print("\nPer-model means:")
    per_model = (df.groupby("model")
                   .agg(rows=("seed", "size"), train_acc=("train_acc", "mean"),
                        ratio=("pos_neg_ratio", "mean")))
    print(per_model.round(3))

    print(f"\nOverall mean train accuracy: {df['train_acc'].mean():.3f} "
          f"-> PASf/iPASa positives:negatives = {df['pos_neg_ratio'].mean():.2f}:1")
    print("iPASwo positives:negatives = 1.00:1 by construction (same wrong items on both sides).")

    if "val_acc" in df and df["val_acc"].notna().any():
        d = df.dropna(subset=["val_acc"])
        if len(d):
            diff = (d["train_acc"] - d["val_acc"]).abs()
            print(f"\n|train_acc - val_acc|: mean {diff.mean():.4f}, max {diff.max():.4f} "
                  f"(validation accuracy is the proxy used before these exact counts existed)")


if __name__ == "__main__":
    main()
