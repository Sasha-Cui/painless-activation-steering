import argparse, yaml, os
from OpenEndedPipeline import OpenEndedPipeline
from dotenv import load_dotenv

# ─── force single‐threaded tokenizers & BLAS ───────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"

# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML experiment file")
    ap.add_argument("--seed", type=int, default=None, help="Override seed from YAML")
    ap.add_argument("--max_sample_size", type=int, help="Override max_sample_size from YAML")
    ap.add_argument("--n_train_proportion", type=float, help="Override n_train_proportion from YAML")
    ap.add_argument("--n_val_proportion", type=float, help="Override n_val_proportion from YAML")
    ap.add_argument("--n_test_proportion", type=float, help="Override n_test_proportion from YAML")
    
    args = ap.parse_args()
    
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    # Override seed if command-line seed is provided
    if args.seed is not None:
        print(f"Overriding YAML seed {params['seed']} with CLI seed {args.seed}")
        params['seed'] = args.seed

    if args.max_sample_size is not None:
        print(f"Overriding YAML max sample size {params['max_sample_size']} with CLI max sample size {args.max_sample_size}")
        params['max_sample_size'] = args.max_sample_size
        
    if args.n_train_proportion is not None:
        print(f"Overriding YAML train prop {params['n_train_proportion']} with CLI train prop {args.n_train_proportion}")
        params['n_train_proportion'] = args.n_train_proportion
    if args.n_val_proportion is not None:
        print(f"Overriding YAML val prop {params['n_val_proportion']} with CLI val prop {args.n_val_proportion}")
        params['n_val_proportion'] = args.n_val_proportion
    if args.n_test_proportion is not None:
        print(f"Overriding YAML test prop {params['n_test_proportion']} with CLI test prop {args.n_test_proportion}")
        params['n_test_proportion'] = args.n_test_proportion
        
    load_dotenv()
    params["hf_token"] = os.getenv("hf_token")
    OpenEndedPipeline(params).run()          # <── one-liner that does everything