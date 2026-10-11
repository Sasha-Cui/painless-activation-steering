"""
OpenEndedPipeline.py
--------------------
Open-ended evaluation of PAS (Appendix H, Table 11).

Steering vectors are built and their layer/strength selected on the multiple-choice
train/validation splits, exactly as in ``BenchmarkSteering.Pipeline``. The test split is
converted to open-ended prompts (answer choices removed); the model generates a free-form
answer with greedy decoding and a GPT-4o grader scores each answer 0/1 against the
reference answer. Requires ``OPENAI_API_KEY`` in the environment (or ``.env``).
"""
from machine_environments import * # Machine Environment Settings; 
import datetime
import traceback  # for printing stack traces on benchmark failures
from preambles import *  # load the preliminary python modules
from benchmark_loading_functions import * # Benchmark loading
from plotting_functions import * # Plotting functions
from load_models_and_data import * # Model loading
from build_steer_vectors_functions import * # Build Steer Vectors Function
from model_prediction import * # Model Prediction Function

class OpenEndedPipeline(object):
    """OpenEndedPipeline of building steering vectors and assessing the performances.

    params
        A dict object denoting the hyperparameters for deployments and building the model architecture.
        See examples under the ``configs`` folder.
    """
    def __init__(self, params):
        super(OpenEndedPipeline, self).__init__()
        self.params = params
        self.start_dt = datetime.now()
        self.timestamp = self.start_dt.strftime("%Y-%m-%d_%H-%M-%S")

        self.seed = params["seed"]
        self.device = params["device"]
        self.device = self.device if torch.cuda.is_available() or self.device != "cuda" else "cpu"

        self.model_name   = params["model_name"]
        self.hf_token     = params["hf_token"]
        self.quantization = params["quantization"]

        self.max_sample_size    = params["max_sample_size"]
        self.n_train_proportion = params["n_train_proportion"]
        self.n_val_proportion   = params["n_val_proportion"]
        self.n_test_proportion  = params["n_test_proportion"]

        # Allow YAML to control this if present
        self.do_train = params.get("do_train", True)

        self.benchmarks      = params["benchmarks"]  # a list of benchmarks
        self.layers_to_try   = sorted(params["layers_to_try"], reverse=True)
        self.steer_strengths = params["steer_strengths"]  # a list of floats
        self.methods         = params["methods"]          # a list of method names (strings)
        self.steer_target    = params["steer_target"]     # "residual", "mlp", "self_attn", or "post_attention_layernorm"

        self.output_dir = (
            f"{params['output_dir']}/"
            f"{self.model_name.replace('/','-')}/open_ended/"
            f"{self.steer_target}/seed_{self.seed}"
        )

    
    def _prepare_logging(self):
        """
        Initialise output folders and redirect stdout / stderr
        to a timestamped logfile.
        """
        self.log_dir     = Path(self.output_dir)
        self.results_dir = Path(self.output_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.vectors_dir = self.results_dir / "steer_vectors"
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.log_dir / f"logfile_{self.timestamp}.txt"
        self._log_fh = open(log_path, "w")

        # Redirect everything
        sys.stdout = sys.stderr = self._log_fh

    def _model_loading(self):
        torch.manual_seed(self.seed)
        real_device = self.device if torch.cuda.is_available() or self.device != "cuda" else "cpu"
        print("\n\n✅Configurations initialised.")

        model, tok = load_lm(
            name=self.model_name,
            device=real_device,
            quantization=self.quantization,
        )
        return model, tok
    def get_config(self):
        """Get the parameters excluding hf_token."""
        safe_config = (yaml.dump({k: ("****" if k in {"hf_token"} else v) for k, v in self.params.items()}, default_flow_style=False))
        return safe_config
    def _set_up(self):
        # ─── make everything reproducible ─────────────────────────
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ────────────────────────────────────────────────────────────
        print(f"[Start] {self.timestamp} - Main function running started")
        login(token=self.hf_token)
        self._prepare_logging()
        print(self.get_config())
        print_env_info()
        model, tok = self._model_loading()
        return model, tok

    def train(self, model, tok):
        print("\n\nLoading benchmarks")

        all_best_steers = {}
        all_test_dfs    = {}

        verbose_on = False
        failed_benchmarks = []

        for bench_name in tqdm(
            self.benchmarks,
            desc="Running Benchmarks",
            disable=not sys.stdout.isatty(),
            leave=False,
        ):
            try:
                print(
                    f"\r→ Running {bench_name}...       ",
                    end="",
                    flush=True,
                )

                if bench_name not in benchmark_map:
                    raise ValueError(f"Benchmark '{bench_name}' not implemented.")

                BenchClass = benchmark_map[bench_name]
                train_df, val_df, test_df, split_meta = load_benchmark(
                    bench_name,
                    max_sample_size=self.max_sample_size,
                    train_prop=self.n_train_proportion,
                    val_prop=self.n_val_proportion,
                    test_prop=self.n_test_proportion,
                    verbose_on=verbose_on,
                    seed=self.seed,
                    open_ended=True,
                )
                all_test_dfs[bench_name] = test_df.copy()

                results = defaultdict(dict)
                steer_vecs_all = {}

                # Baseline (unsteered) MC accuracy on VAL
                train_preds_un = pd.Series(
                    predict_mcq_letters(train_df, model, tok, device=self.device),
                    index=train_df.index,
                )

                raw_acc = evaluate_with_steering(
                    model=model,
                    tok=tok,
                    df=val_df,
                    steer=None,
                    steer_strength=1.0,
                    layer=15,
                    steer_target=self.steer_target,
                    device=self.device,
                )

                print(f"\n✅ Baseline (Unsteered) Accuracy on the Validation Set: {raw_acc:.2%}")

                train_predictions = train_preds_un

                # ─── build steering vectors ─────────────────────────
                for layer in self.layers_to_try:
                    print(f"\n🔍 Layer {layer}")

                    builder = SteeringVectorBuilder(
                        model=model,
                        tok=tok,
                        layer=layer,
                        steer_target=self.steer_target,
                        token_mode="last",
                        device=self.device,
                    )

                    steer_vecs = {}
                    builder_dispatch = {
                        "BAS_full_mcq":       builder.build_BAS_full_mcq,
                        "iBAS_all":           builder.build_iBAS_all,
                        "iBAS_wrong_only":    builder.build_iBAS_wrong_only,
                    }

                    for method in self.methods:
                        fn = builder_dispatch[method]
                        vec = fn(train_df, train_predictions)
                        steer_vecs[method] = vec
                        steer_vecs_all[(layer, method)] = vec

                    bench_vec_dir = self.vectors_dir / bench_name
                    bench_vec_dir.mkdir(parents=True, exist_ok=True)

                    for method, vec in steer_vecs.items():
                        vec_cpu = vec.detach().cpu()
                        for strength in self.steer_strengths:
                            fname_all = f"{method}_layer{layer}_strength{strength}.pt"
                            torch.save(vec_cpu, bench_vec_dir / fname_all)

                    for strength in self.steer_strengths:
                        accs = {
                            m: evaluate_with_steering(
                                model=model,
                                tok=tok,
                                df=val_df,
                                steer=steer_vecs[m],
                                steer_strength=strength,
                                layer=layer,
                                steer_target=self.steer_target,
                                device=self.device,
                            )
                            for m in self.methods
                        }
                        results[(layer, strength)] = accs

                        print(f"  🔧 Steering strength = {strength} sweep completed.")
                        if verbose_on:
                            for method, acc in accs.items():
                                delta = acc - raw_acc
                                print(f"    {method:25s}: {acc:.2%} (Δ {delta:+.3f})")

                plot_delta_heatmaps(
                    results=results,
                    raw_acc=raw_acc,
                    save_dir=self.results_dir / "figures",
                    benchmark_name=bench_name,
                    model_name=self.model_name,
                    methods=self.methods,
                )
                plot_results(
                    results=results,
                    raw_acc=raw_acc,
                    save_dir=self.results_dir / "tables",
                    benchmark_name=bench_name,
                    model_name=self.model_name,
                    methods=self.methods,
                )

                # ─── Best configs from VAL ─────────────────────────
                best_cfg = {method: {"acc": -float("inf")} for method in accs.keys()}

                for (layer, strength), method_accs in results.items():
                    for method, acc in method_accs.items():
                        if acc > best_cfg[method]["acc"]:
                            best_cfg[method] = {
                                "acc":      acc,
                                "layer":    layer,
                                "strength": strength,
                            }

                print(f"\n🔝 Best Hyperparameters for {bench_name} selected from validation set:")
                for method, cfg_obj in best_cfg.items():
                    print(
                        f"  {method:25s}: "
                        f"layer={cfg_obj['layer']:2}, "
                        f"strength={cfg_obj['strength']}, "
                        f"val_acc={cfg_obj['acc']:.2%}"
                    )

                # Extract the actual best steering vectors
                best_steers_for_bench = {}
                for method, cfg_obj in best_cfg.items():
                    best_layer    = cfg_obj["layer"]
                    best_strength = cfg_obj["strength"]
                    best_acc      = cfg_obj["acc"]

                    best_vec = steer_vecs_all[(best_layer, method)]

                    best_steers_for_bench[method] = {
                        "vector":   best_vec,
                        "layer":    best_layer,
                        "strength": best_strength,
                        "val_acc":  best_acc,
                    }

                all_best_steers[bench_name] = best_steers_for_bench

            except Exception as e:
                err_msg = f"\n❌ Exception while processing benchmark '{bench_name}': {e}"
                print(err_msg)
                traceback.print_exc()
                failed_benchmarks.append(bench_name)
                continue

        return all_test_dfs, all_best_steers

    def run(self):
        model, tok = self._set_up()
        if not self.do_train:
            return None, None, None, None

        all_test_dfs, all_best_steers = self.train(model, tok)

        # {benchmark: {method: mean_score}}
        all_test_scores = {}

        for bench_name, test_df in all_test_dfs.items():
            best_steers_for_bench = all_best_steers[bench_name]
            all_test_scores[bench_name] = {}

            # ---- Unsteered baseline on TEST ----
            baseline_save_dir = (
                self.results_dir / "predictions" / bench_name / "raw_model"
            )

            baseline_test_score, baseline_preds = evaluate_open_ended_with_steering(
                model=model,
                tok=tok,
                df=test_df,
                steer=None,              # unsteered
                steer_strength=1.0,      # irrelevant when steer=None
                layer=15,                # unused when steer=None
                steer_target=self.steer_target,
                device=self.device,
                return_preds=True,
                save_dir=baseline_save_dir,
            )
            all_test_scores[bench_name]["raw_model"] = baseline_test_score
            print(f"\n[{bench_name}] raw_model (unsteered) mean score on TEST: {baseline_test_score:.3f}")

            # ---- Steered evaluations using best configs ----
            for method, cfg in best_steers_for_bench.items():
                best_vec      = cfg["vector"]
                best_layer    = cfg["layer"]
                best_strength = cfg["strength"]

                save_dir = (
                    self.results_dir / "predictions" / bench_name / method
                )

                test_score, steered_preds = evaluate_open_ended_with_steering(
                    model=model,
                    tok=tok,
                    df=test_df,
                    steer=best_vec,
                    steer_strength=best_strength,
                    layer=best_layer,
                    steer_target=self.steer_target,
                    device=self.device,
                    return_preds=True,
                    save_dir=save_dir,
                )

                all_test_scores[bench_name][method] = test_score
                delta = test_score - baseline_test_score

                print(
                    f"  • {method:25s}: "
                    f"test_score = {test_score:.3f} "
                    f"(Δ vs raw_model = {delta:+.3f})"
                )

                # ---- Save method-level results into a per-benchmark CSV ----
                bench_summary_dir = self.results_dir / "tables" / bench_name
                bench_summary_dir.mkdir(parents=True, exist_ok=True)
                
                summary_path = bench_summary_dir / "open_ended_method_results.csv"
                
                # One row per method
                row = {
                    "benchmark": bench_name,
                    "method": method,
                    "test_score": round(test_score, 6),
                    "delta_vs_raw": round(delta, 6),
                    "layer": best_layer,
                    "strength": best_strength,
                }
                
                # Append or create CSV
                if summary_path.exists():
                    prev = pd.read_csv(summary_path)
                    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
                    prev.to_csv(summary_path, index=False)
                else:
                    pd.DataFrame([row]).to_csv(summary_path, index=False)


        # ─── Build single table: rows = benchmarks, cols = methods ────────
        all_methods = set()
        for scores in all_test_scores.values():
            all_methods.update(scores.keys())

        all_methods = ["raw_model"] + sorted(m for m in all_methods if m != "raw_model")

        benchmarks_sorted = sorted(all_test_scores.keys())
        score_table = pd.DataFrame(index=benchmarks_sorted, columns=all_methods, dtype=float)

        for bench_name, scores in all_test_scores.items():
            for method_name, score in scores.items():
                score_table.loc[bench_name, method_name] = score

        tables_dir = self.results_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / "open_ended_test_scores.csv"
        score_table.to_csv(table_path, index=True)

        print(f"\n✅ Saved open-ended test score table to: {table_path}")

        return score_table, all_test_scores, all_test_dfs, all_best_steers
