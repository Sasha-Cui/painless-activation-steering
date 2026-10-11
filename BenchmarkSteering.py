"""
BenchmarkSteering.py
--------------------
Single-file implementation of the whole pipeline.

The class ``Pipeline`` wraps:
  • environment setup
  • model & tokenizer loading (4-/8-/no-bit)
  • benchmark loading/splitting
  • steering-vector construction (five methods)
  • exhaustive layer × strength grid search
  • test-set evaluation with best hyper-params
  • result visualisation & logging
"""
from machine_environments import * # Machine Environment Settings; 
import dateutil.tz
import datetime
import traceback  # for printing stack traces on benchmark failures
from preambles import *  # load the preliminary python modules
from benchmark_loading_functions import * # Benchmark loading
from plotting_functions import * # Plotting functions
from load_models_and_data import * # Model loading
from build_steer_vectors_functions import * # Build Steer Vectors Function
from model_prediction import * # Model Prediction Function


class Pipeline(object):
    """Pipeline of building steering vectors and assessing the performances.

    params
        A dict object denoting the hyperparameters for deployments and building the model architecture.
        See examples under the ``configs`` folder.
    """
    def __init__(self, params):
        super(Pipeline, self).__init__()
        self.params = params
        self.start_dt = datetime.now()
        self.timestamp = self.start_dt.strftime("%Y-%m-%d_%H-%M-%S")
        self.seed = params['seed']
        self.device = params['device']
        self.device = self.device if torch.cuda.is_available() or self.device != "cuda" else "cpu"
        self.model_name = params['model_name']
        self.hf_token = params['hf_token']
        self.quantization = params['quantization']

        if "prompt_pairs" in params:
            self.prompt_pairs = (
                params["prompt_pairs"]
                if isinstance(params["prompt_pairs"], Path)
                else Path(params["prompt_pairs"])
            )

        self.max_sample_size = params["max_sample_size"]
        self.n_train_proportion = params["n_train_proportion"]
        self.n_val_proportion = params["n_val_proportion"]
        self.n_test_proportion = params["n_test_proportion"]

        self.benchmarks = params['benchmarks'] # a list of benchmarks
        self.layers_to_try = sorted(params['layers_to_try'], reverse=True)
        self.steer_strengths = params['steer_strengths'] # a list of integers
        self.methods = params['methods'] # a list of integers
        self.steer_target = params["steer_target"] # "residual", "mlp", "self_attn", or "post_attention_layernorm"
        self.icl_k = int(params.get("icl_k", 0))
        
        val = params.get("catastrophic_test", False)
        self.catastrophic_test = val
        self.target_benchmark = params.get("target_benchmark", None) # the benchmark for testing catastrophic forgetting. It is used only when self.catastrophic_test = True
        if self.catastrophic_test:
            if not self.target_benchmark:
                raise ValueError("`target_benchmark` must be set when `catastrophic_test` is True.")
            self.output_dir = (
                f"{params['output_dir']}/{self.model_name.replace('/','-')}"
                f"/catastrophic_forgetting/seed_{self.seed}"
            )
        elif self.icl_k > 0:
            self.output_dir = (
                f"{params['output_dir']}/{self.model_name.replace('/','-')}"
                f"/ICL/seed_{self.seed}"
            )
        else:
            self.output_dir = f"{params['output_dir']}/{self.model_name.replace('/','-')}/{self.steer_target}/seed_{self.seed}"
        

    
    def run(self):
        # ─── make everything reproducible ─────────────────────────
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        # force deterministic cuDNN (may slow things down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # ────────────────────────────────────────────────────────────

        
        print(f"[Start] {self.timestamp} - Main function running started")
        if self.hf_token:  # otherwise fall back to the cached CLI login (~/.cache/huggingface/token)
            login(token=self.hf_token)
        self._prepare_logging()
        print(self.get_config())
        print_env_info()
        model, tok = self._model_loading()

        if "handcrafted_prompt_pair" in self.methods:
            pairs = self._load_prompt_pairs()
            self._run_all_benchmarks(model=model, tok=tok, pairs=pairs)
        else:
            self._run_all_benchmarks(model=model, tok=tok)

        end_time = datetime.now()
        print(f"[End] {end_time.strftime('%Y-%m-%d_%H_%M_%S')} - Main function running completed - [Elapsed] {str(end_time - self.start_dt)}")


    def get_config(self):
        """Get the parameters excluding hf_token."""
        safe_config = (yaml.dump({k: ("****" if k in {"hf_token", "openai_api_key"} else v) for k, v in self.params.items()}, default_flow_style=False))
        return safe_config


    def _prepare_logging(self):
        """
        Initialise output folders and redirect stdout / stderr
        to a timestamped logfile.

        Side-effects
        ------------
        • self.log_dir       →  <output_dir>/log
        • self.results_dir   →  <output_dir>/results
        • self._log_fh       →  open file handle (so you can close it later)
        • sys.stdout / sys.stderr are redirected to that file
        """
        #  Create <output_dir>/log  and  <output_dir>/results
        self.log_dir     = Path(self.output_dir)
        self.results_dir = Path(self.output_dir)
        self.log_dir.mkdir(parents=True,     exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.vectors_dir = self.results_dir / "steer_vectors"
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        # Open logfile with timestamp so runs don't overwrite each other
        log_path = self.log_dir / f"logfile_{self.timestamp}.txt"
        self._log_fh = open(log_path, "w")
        sys.stdout = sys.stderr = self._log_fh

    def _model_loading(self):
        torch.manual_seed(self.seed)
        real_device = self.device if torch.cuda.is_available() or self.device != "cuda" else "cpu"
        print("\n\n✅Configurations initialised.")

        # Load the model
        model, tok = load_lm(name = self.model_name, device=real_device, quantization = self.quantization)
        return model, tok

    def _load_prompt_pairs(self):
        """Load CAA prompt pairs, grouped by the task they were written for.

        Returns {task: [(positive, negative), ...]}. The steering vector for a
        benchmark is built only from that benchmark's pairs, as in CAA
        (Rimsky et al., 2024), where each behavior has its own vector. A file
        without a `task` column falls back to a single pooled set.
        """
        print(f"\n\n>>> Loading CAA prompt pairs from {self.prompt_pairs}")
        df = pd.read_csv(self.prompt_pairs)
        if "task" not in df.columns:
            pairs = [
                (str(pos).strip(), str(neg).strip())
                for pos, neg in zip(df["positive_example"], df["negative_example"])
                if isinstance(pos, str) and isinstance(neg, str)
            ]
            print(f"Loaded {len(pairs)} prompt pairs (no `task` column: pooled across tasks)")
            return {None: pairs}

        pairs_by_task = {}
        for task, sub_df in df.groupby("task"):
            pairs_by_task[task] = [
                (str(pos).strip(), str(neg).strip())
                for pos, neg in zip(sub_df["positive_example"], sub_df["negative_example"])
                if isinstance(pos, str) and isinstance(neg, str)
            ]
        total = sum(len(v) for v in pairs_by_task.values())
        print(f"Loaded {total} prompt pairs across {len(pairs_by_task)} tasks: "
              f"{ {k: len(v) for k, v in sorted(pairs_by_task.items())} }")
        return pairs_by_task

    def _pairs_for_benchmark(self, pairs, bench_name):
        """Select the pairs written for `bench_name` (or the pooled set)."""
        if not isinstance(pairs, dict):
            return pairs
        if None in pairs:
            return pairs[None]
        if bench_name not in pairs:
            raise ValueError(
                f"No prompt pairs for benchmark '{bench_name}' in {self.prompt_pairs}; "
                f"available tasks: {sorted(pairs)}"
            )
        return pairs[bench_name]

    def _run_all_benchmarks(self, model, tok, pairs=None):
        print("\n\nLoading benchmarks")
        # Store results across all benchmarks
        all_benchmark_results = {}
        all_best_configs = {}
        all_test_results = {}
        all_flip_counts = {}     
        verbose_on = False
        failed_benchmarks = []           # keep track of any benchmarks that error out
        if self.catastrophic_test:
            _, _, _target_test_df, _ = load_benchmark(
                self.target_benchmark,
                max_sample_size=self.max_sample_size,
                train_prop=self.n_train_proportion,
                val_prop=self.n_val_proportion,
                test_prop=self.n_test_proportion,
                verbose_on=verbose_on,
                seed=self.seed
            )
        for bench_name in tqdm(
            self.benchmarks,
            desc="Running Benchmarks",
            disable=not sys.stdout.isatty(),
            leave=False,
            file=sys.__stdout__,
        ):
            try:
                print(
                    f"\r→ Running {bench_name}...       ",
                    end="",
                    file=sys.__stdout__,
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
                    seed = self.seed
                )

                if self.catastrophic_test:
                    test_df = _target_test_df
                
                # Initialize results for this benchmark
                results = defaultdict(dict)
                # Baseline accuracy
                train_preds_un = pd.Series(
                        predict_mcq_letters(train_df, model, tok, device=self.device),
                        index=train_df.index,
                    )
                # Build the ICL exemplar pool from WRONG-TRAIN-BEFORE-ICL
                wrong_train_df0 = _wrong_rows(train_df, train_preds_un) if self.icl_k > 0 else None
                
                # Compute baseline on VAL
                if self.icl_k == 0:
                    # unsteered baseline
                    raw_acc = evaluate_with_steering(
                        model=model, tok=tok, df=val_df, steer=None,
                        steer_strength=1.0, layer=15, steer_target=self.steer_target,
                        device=self.device
                    )
                else:
                    # new behavior: baseline = ICL-only on VAL (no steering), using the same pool from TRAIN
                    raw_acc = evaluate_with_steering_ICL(
                        model=model, tok=tok, df_test=val_df,
                        wrong_train_df=wrong_train_df0, k=self.icl_k,
                        steer=None, steer_strength=0.0, layer=15,
                        steer_target=self.steer_target, device=self.device
                    )

                print(f"\n✅ Baseline ({'ICL' if self.icl_k>0 else 'Unsteered'}) Accuracy on the Validation Set: {raw_acc:.2%}")

                # TRAIN pass WITH ICL to identify wrong-after-ICL for vector building
                if self.icl_k > 0:
                    _, train_preds_icl = evaluate_with_steering_ICL(
                        model=model, tok=tok, df_test=train_df,
                        wrong_train_df=wrong_train_df0, k=self.icl_k,
                        steer=None, steer_strength=0.0, layer=15,
                        steer_target=self.steer_target, device=self.device,
                        return_preds=True,
                        save_dir=None
                    )
                    # We will build vectors based on the *ICL predictions*:
                    train_predictions = train_preds_icl
                else:
                    # No ICL: keep the old behavior
                    train_predictions = train_preds_un
                
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

                    steer_vecs = {} # container for all vectors produced at this layer

                    # ----- handcrafted prompt-pair--------------
                    if "handcrafted_prompt_pair" in self.methods:
                        steer_vecs["handcrafted_prompt_pair"] = (
                            build_steer_vec_handcrafted_prompt_pair(
                                model=model,
                                tok=tok,
                                pairs=self._pairs_for_benchmark(pairs, bench_name),
                                layer=layer,
                                device=self.device,
                                batch=64,
                                token_mode="last",
                            )
                        )


                    builder_dispatch = {
                        "BAS_full_mcq":         builder.build_BAS_full_mcq,
                        "iBAS_all":             builder.build_iBAS_all,
                        "iBAS_wrong_only":      builder.build_iBAS_wrong_only,
                    }

                    for method in self.methods:
                        if method == "handcrafted_prompt_pair":
                            continue                    # already handled above
                        fn = builder_dispatch[method]   # look up the right builder method
                        steer_vecs[method] = fn(train_df, train_predictions)

                    bench_vec_dir = (self.vectors_dir / bench_name)
                    bench_vec_dir.mkdir(parents=True, exist_ok=True)

                    for method, vec in steer_vecs.items():
                        vec_cpu = vec.detach().cpu()
                        for strength in self.steer_strengths:
                            fname_all = f"{method}_layer{layer}_strength{strength}.pt"
                            torch.save(vec_cpu, bench_vec_dir / fname_all)
                        
                    for strength in self.steer_strengths:
                        if self.icl_k == 0:
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
                        else:
                            accs = {
                                m: evaluate_with_steering_ICL(
                                    model=model, tok=tok, df_test=val_df,
                                    wrong_train_df=wrong_train_df0, k=self.icl_k,
                                    steer=steer_vecs[m], steer_strength=strength,
                                    layer=layer, steer_target=self.steer_target,
                                    device=self.device
                                )
                                for m in self.methods
                            }
                        results[(layer, strength)] = accs
                        # Log
                        print(f"  🔧 Steering strength = {strength} sweep completed.")
                        if verbose_on:
                            for method, acc in accs.items():
                                delta = acc - raw_acc
                                print(f"    {method:25s}: {acc:.2%} (Δ {delta:+.3f})")

                plot_delta_heatmaps(results=results, raw_acc = raw_acc, save_dir=self.results_dir/"figures", benchmark_name = bench_name, model_name=self.model_name, methods=self.methods)
                plot_results(results=results, raw_acc = raw_acc, save_dir=self.results_dir/"tables", benchmark_name = bench_name, model_name=self.model_name, methods=self.methods)

                # ─────────────────────────────── Best Configurations ──────────────────────────────

                best_cfg = {method: {"acc": -float("inf")} for method in accs.keys()}

                for (layer, strength), method_accs in results.items():
                    for method, acc in method_accs.items():
                        if acc > best_cfg[method]["acc"]:
                            best_cfg[method] = {"acc": acc, "layer": layer, "strength": strength}

                print(f"\n🔝 Best Hyperparameters for {bench_name} selected from validation set:")
                for method, cfg_obj in best_cfg.items():
                    print(f"  {method:25s}: layer={cfg_obj['layer']:2}, strength={cfg_obj['strength']}, val_acc={cfg_obj['acc']:.2%}")


                # ─────────────────────────────── Evaluate Methods in the Test Set ──────────────────────────────

                # Rebuild best steering vectors for each method
                print("\nEvaluating the best configurations on the test set.")
                print("Rebuild best steering vectors for each method for test set evaluation.")
                steer_vectors = {}

                for method, cfg_obj in best_cfg.items():
                    layer = cfg_obj["layer"]

                    if method == "handcrafted_prompt_pair":
                        vec = build_steer_vec_handcrafted_prompt_pair(
                            model=model,
                            tok=tok,
                            pairs=self._pairs_for_benchmark(pairs, bench_name),
                            layer=layer,
                            steer_target=self.steer_target,
                            device=self.device,
                            batch=64,
                            token_mode="last"
                        )

                    else:
                        builder = SteeringVectorBuilder(
                            model=model,
                            tok=tok,
                            layer=layer,
                            steer_target=self.steer_target,
                            token_mode="last",
                            device=self.device
                        )

                        if method == "BAS_full_mcq":
                            vec = builder.build_BAS_full_mcq(train_df, train_predictions)
                        elif method == "iBAS_all":
                            vec = builder.build_iBAS_all(train_df, train_predictions)
                        elif method == "iBAS_wrong_only":
                            vec = builder.build_iBAS_wrong_only(train_df, train_predictions)
                        else:
                            raise ValueError(f"Unknown method: {method}")

                    steer_vectors[method] = vec

                # Evaluate raw baseline (unsteered). If ICL>0, evaluate with ICL on wrong-train exemplars.
                if self.icl_k == 0:
                    raw_test_acc, unsteered_preds = evaluate_with_steering(
                        model=model,
                        tok=tok,
                        df=test_df,
                        steer=None,
                        steer_strength=1.0,
                        layer=15,  # any layer; no steering
                        steer_target=self.steer_target,
                        device=self.device,
                        return_preds=True,
                        save_dir=self.results_dir / "predictions" / bench_name / "Unsteered"
                    )
                else:
                    raw_test_acc, unsteered_preds = evaluate_with_steering_ICL(
                        model=model,
                        tok=tok,
                        df_test=test_df,
                        wrong_train_df=wrong_train_df0,   # built above
                        k=self.icl_k,
                        steer=None,                      # unsteered + ICL
                        steer_strength=0.0,
                        layer=15,                        # unused when steer=None but kept for symmetry
                        steer_target=self.steer_target,
                        device=self.device,
                        return_preds=True,
                        save_dir=self.results_dir / "predictions" / bench_name / f"Unsteered_ICL{kself.icl_k if False else ''}".replace("kself.", f"k{self.icl_k}")
                    )


                #  ─────────────────── store the rebuilt tensors ───────────────────
                for method, vec in steer_vectors.items():
                    vec_cpu = vec.detach().cpu()
                    layer = best_cfg[method]["layer"]
                    strength = best_cfg[method]["strength"]
                    fname = f"{method}_layer{layer}_strength{strength}.pt"
                    vec_path = (self.vectors_dir / bench_name / fname)
                    vec_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(vec_cpu, vec_path)
                    fname_best = f"{method}_layer{layer}_strength_best.pt"
                    torch.save(vec_cpu, (self.vectors_dir / bench_name / fname_best))


                # Evaluate all methods using their best parameters
                test_accs = {}

                for method in best_cfg:
                    cfg_obj = best_cfg[method]
                    if self.icl_k == 0:
                        acc, steered_preds = evaluate_with_steering(
                            model=model,
                            tok=tok,
                            df=test_df,
                            steer=steer_vectors[method],
                            steer_strength=cfg_obj["strength"],
                            layer=cfg_obj["layer"],
                            steer_target=self.steer_target,
                            device=self.device,
                            return_preds=True,
                            save_dir=self.results_dir / "predictions" / bench_name / method
                        )
                    else:
                        acc, steered_preds = evaluate_with_steering_ICL(
                            model=model,
                            tok=tok,
                            df_test=test_df,
                            wrong_train_df=wrong_train_df0,     # same wrong-train pool
                            k=self.icl_k,
                            steer=steer_vectors[method],       # steered + ICL
                            steer_strength=cfg_obj["strength"],
                            layer=cfg_obj["layer"],
                            steer_target=self.steer_target,
                            device=self.device,
                            return_preds=True,
                            save_dir=self.results_dir / "predictions" / bench_name / f"{method}_ICLk{self.icl_k}"
                        )
                    test_accs[method] = acc

                    # ----------- divergence bookkeeping -----------
                    r2w, w2r = compare_predictions(test_df,
                                                   unsteered_preds,
                                                   steered_preds)

                    diff_dir = (self.results_dir / "predictions" / bench_name /
                                f"{method}_diff")
                    diff_dir.mkdir(parents=True, exist_ok=True)
                    r2w.to_csv(diff_dir / "R2W.csv", index=False)
                    w2r.to_csv(diff_dir / "W2R.csv", index=False)

                    # store counts for summary
                    all_flip_counts.setdefault(bench_name, {})[method] = {
                        "R2W": len(r2w),
                        "W2R": len(w2r)
                    }

                    print(f"    ↪ {method:25s}: R2W={len(r2w):3d}   W2R={len(w2r):3d}")

                # Generate test set visualization for this benchmark
                plot_bar(
                    results={"Unsteered": raw_test_acc, **test_accs},
                    title=f"{bench_name} Steering Accuracy on Test Set",
                    save_path=f"{self.results_dir}/figures/{bench_name}/test_accuracy_comparison.png",
                )

                # Store results for cross-benchmark analysis
                all_benchmark_results[bench_name] = {
                    'val_results': results,
                    'raw_val_acc': raw_acc,
                    'raw_test_acc': raw_test_acc
                }
                all_best_configs[bench_name] = best_cfg
                all_test_results[bench_name] = {"Unsteered": raw_test_acc, **test_accs}

            except Exception as e:
                # Ensure traceback is printed to both the logfile and real stdout for visibility
                err_msg = f"\n❌ Exception while processing benchmark '{bench_name}': {e}"
                print(err_msg)
                traceback.print_exc()
                failed_benchmarks.append(bench_name)
                continue  # proceed with the next benchmark

        # ─────────────────────────────── Cross-Benchmark Analysis ──────────────────────────────
        print(f"\n\n{'='*80}")
        print("📊 CROSS-BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print("🔝 Best Configurations:")

        # Collect structured rows
        rows = []
        for bench_name in all_best_configs:
            best_cfg = all_best_configs[bench_name]
            test_results = all_test_results[bench_name]

            # ←── add raw (unsteered) accuracies
            rows.append({
                "Benchmark": bench_name,
                "Method":    "Unsteered",
                "Layer":     None,
                "Strength":  None,
                "Val-Acc":   all_benchmark_results[bench_name]["raw_val_acc"],
                "Test-Acc":  test_results["Unsteered"]
            })

            for method, cfg_obj in best_cfg.items():
                test_acc = test_results[method]
                rows.append({
                    "Benchmark": bench_name,
                    "Method": method,
                    "Layer": cfg_obj["layer"],
                    "Strength": cfg_obj["strength"],
                    "Val-Acc": cfg_obj["acc"],
                    "Test-Acc": test_acc
                })

        # Format accuracy columns as percentages for display (but save raw float in CSV)
        df = pd.DataFrame(rows)
        df_display = df.copy()
        df_display["Val-Acc"] = df_display["Val-Acc"].apply(lambda x: f"{x:.2%}")
        df_display["Test-Acc"] = df_display["Test-Acc"].apply(lambda x: f"{x:.2%}")

        # Print nicely
        print(df_display.to_string(index=False))

        # Save raw (unformatted) values to CSV
        csv_path = self.results_dir / "tables" / "cross_benchmark_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Saved summary CSV to {csv_path}")

        # ------------------------------------------------------------------
        #  Flip-count summary
        # ------------------------------------------------------------------
        flip_rows = []
        for bench_name, m_dict in all_flip_counts.items():
            for method, cnts in m_dict.items():
                flip_rows.append({
                    "Benchmark": bench_name,
                    "Method":    method,
                    "R2W":       cnts["R2W"],
                    "W2R":       cnts["W2R"]
                })

        if flip_rows:          # only if at least one benchmark produced data
            df_flips = pd.DataFrame(flip_rows)
            flip_csv_path = self.results_dir / "tables" / "cross_benchmark_flips.csv"
            df_flips.to_csv(flip_csv_path, index=False)
            print(f"\n📄 Saved flip-count CSV to {flip_csv_path}")


        # Generate plots
        if len(all_benchmark_results) > 1:
            plot_cross_benchmark_comparison(
                all_test_results,
                all_best_configs,
                save_dir=self.results_dir / "figures",
                model_name=self.model_name
            )
        print(f"\n✅ Multi-benchmark analysis complete. Processed {len(all_benchmark_results)} benchmarks.")

        if failed_benchmarks:
            print("\n⚠️  The following benchmarks failed and were skipped due to errors:")
            for b in failed_benchmarks:
                print(f"   • {b}")
        else:
            print("\n✅ All benchmarks completed without uncaught exceptions.")

        sys.stdout = sys.__stdout__   # Restore console output
        sys.stderr = sys.__stderr__   # Also restore stderr
        self._log_fh.close()          # Close the file handle correctly
