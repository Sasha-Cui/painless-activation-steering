#!/usr/bin/env python
"""
generate_caa_pairs.py
---------------------
Generate CAA contrastive prompt pairs with a model, for the tasks listed in a
generator-prompt CSV (columns: task, prompt), and append them to that model's
pair file.

Mirrors the procedure used for the 15 behavior tasks (CAA_baseline_generation.ipynb):
the generator prompt is wrapped in the model's chat template with the system message
"You are a helpful AI assistant." and decoded greedily, then the generation is parsed
into (positive, negative) pairs.

Example:
    python generate_caa_pairs.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --generators contrastive_prompt_generators_intelligence.csv \
        --out prompt_pairs_outputs/DeepSeek-R1-Distill-Llama-8B_contrastive_pairs_expanded.csv \
        --raw_out prompt_pairs_outputs/raw/DeepSeek-R1-Distill-Llama-8B_contrastive_pairs_intelligence.csv

Rows for tasks already present in --out are left untouched; new tasks are appended.
"""
from machine_environments import *   # cache settings
from preambles import *              # torch, pandas, os, sys, ...
from load_models_and_data import load_lm

import argparse
import csv
import re
from dotenv import load_dotenv

FIELDS = ["model_name", "task", "pair_id", "positive_example", "negative_example", "generator_prompt"]

# Accepts "1. **Positive:**", "### Pair 1 / **Positive:**", "- **Positive**:", etc.
MARK = re.compile(r'(?i)(?:(?<=^)|(?<=\n)|(?<=[.!?"\u201d]\s)|(?<=[.!?"\u201d]\s\s))[ \t]*(?:\d+[\.\)]\s*)?(?:[#>\-\*\s]*)\*{0,2}(positive|negative)\*{0,2}(?:\s+story)?\*{0,2}\s*[:：]\s*', re.M)
SEP = re.compile(r'(?m)^\s*(?:---+|###|\*\*Pair|\d+[\.\)]\s|Pair\s+\d+)')


def clean(s):
    return re.sub(r'\s+', ' ', s).strip(' *_-"“”')


def parse_pairs(text, max_pairs=10, minlen=20):
    """Split one generation into (positive, negative) pairs."""
    body = text.split('</think>', 1)[1] if '</think>' in text else text   # drop chain of thought
    marks = [(m.start(), m.end(), m.group(1).lower()) for m in MARK.finditer(body)]
    seqs = []
    for i, (s, e, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        seg = body[e:end]
        cut = SEP.search(seg)
        if cut:
            seg = seg[:cut.start()]
        seqs.append((kind, clean(seg)))
    pairs, i = [], 0
    while i < len(seqs) - 1:
        if seqs[i][0] == 'positive' and seqs[i + 1][0] == 'negative':
            p, n = seqs[i][1], seqs[i + 1][1]
            if len(p) >= minlen and len(n) >= minlen:
                pairs.append((p, n))
            i += 2
        else:
            i += 1
    return pairs[:max_pairs]


def build_chat_input(tok, user_prompt):
    messages = [{"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}]
    if getattr(tok, "apply_chat_template", None) is not None:
        return tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    text = "You are a helpful AI assistant.\n\n" + user_prompt
    return tok(text, return_tensors="pt")["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--generators", required=True, help="CSV with columns task,prompt")
    ap.add_argument("--out", required=True, help="pair CSV to append to (created if absent)")
    ap.add_argument("--raw_out", default=None, help="optional CSV of raw generations")
    ap.add_argument("--max_new_tokens", type=int, default=10240)
    ap.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    load_dotenv()
    if os.getenv("hf_token"):
        login(token=os.getenv("hf_token"))

    gens = list(csv.DictReader(open(args.generators)))
    short = args.model.split("/")[-1]

    existing, have = [], set()
    if os.path.exists(args.out):
        existing = list(csv.DictReader(open(args.out)))
        have = {r["task"] for r in existing}
        print(f">>> {args.out} already has {len(existing)} rows for {len(have)} tasks", flush=True)
    todo = [g for g in gens if g["task"] not in have]
    if not todo:
        print(">>> nothing to generate; all tasks already present.", flush=True)
        return

    print(f">>> generating pairs for: {[g['task'] for g in todo]}", flush=True)
    model, tok = load_lm(name=args.model, device=args.device, quantization=args.quantization)
    model.eval()

    new_rows, raw_rows = [], []
    for g in todo:
        print(f"    [{g['task']}] generating ...", flush=True)
        ids = build_chat_input(tok, g["prompt"]).to(args.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        raw_rows.append({"model_name": short, "task": g["task"], "generator_prompt": g["prompt"], "model_output": text})
        pairs = parse_pairs(text)
        print(f"    [{g['task']}] parsed {len(pairs)} pairs", flush=True)
        if len(pairs) < 10:
            print(f"    WARNING: only {len(pairs)} pairs parsed for {g['task']}", flush=True)
        for i, (p, n) in enumerate(pairs, 1):
            new_rows.append({"model_name": short, "task": g["task"], "pair_id": i,
                             "positive_example": p, "negative_example": n, "generator_prompt": g["prompt"]})

    if args.raw_out:
        os.makedirs(os.path.dirname(args.raw_out), exist_ok=True)
        with open(args.raw_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model_name", "task", "generator_prompt", "model_output"])
            w.writeheader(); w.writerows(raw_rows)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        w.writerows(new_rows)
    print(f">>> appended {len(new_rows)} pairs to {args.out} "
          f"(now {len(existing) + len(new_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
