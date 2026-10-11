from preambles import *
import openai
from util import get_layer
# ─────────────────────────────── Building Model Predictions on a Data Frame ──────────────────────────────

def predict_mcq_letters(df, model, tok, device="cuda"):
    model.eval()
    predictions = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Model is Generating MCQ Predictions", disable=not sys.stdout.isatty(), leave=False, file = sys.__stdout__):
        prompt = row["full_prompt"]
        n_choices = row["num_choices"]
        valid_letters = list(ascii_uppercase[:n_choices])
        valid_token_ids = tok.convert_tokens_to_ids(valid_letters)

        # Defensive: convert to list if a single int (e.g., when only one choice)
        if isinstance(valid_token_ids, int):
            valid_token_ids = [valid_token_ids]

        # Tokenize and forward pass
        device = next(model.parameters()).device
        enc = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            out = model(**enc)
            logits = out.logits[:, -1, :]  # Predict next token

        # Get most probable valid letter
        choice_logits = logits[0, valid_token_ids]
        pred_index = torch.argmax(choice_logits).item()
        pred_letter = valid_letters[pred_index]
        predictions.append(pred_letter)

    return predictions  # list of predicted letter choices


# ─────────────────────────────── Performance Evaluation ──────────────────────────────
@contextmanager
def apply_steering_hook(
    model,
    layer_index: int,
    steer: torch.Tensor,
    steer_strength: float,
    steer_target: Literal["residual", "mlp", "self_attn", "post_attn"] = "residual"
):
    if steer_target == "post_attn":
        steer_target = "post_attention_layernorm"
    def hook_fn(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        actual_output = output[0] if is_tuple else output

        if not isinstance(actual_output, torch.Tensor):
            raise TypeError(f"Unexpected output type: {type(actual_output)}")

        if actual_output.ndim != 3 or actual_output.shape[-1] != steer.shape[-1]:
            raise ValueError(f"Shape mismatch: output {actual_output.shape}, steer {steer.shape}")

        modified_output = actual_output.clone()
        modified_output[:, -1, :] += steer.to(actual_output.device, dtype=actual_output.dtype) * steer_strength

        if is_tuple:
            return (modified_output,) + output[1:]
        return modified_output

    layer = get_layer(model, layer_index)

    if steer_target == "residual":
        target = layer
    else:
        target = getattr(layer, steer_target, None)
        if target is None:
            raise ValueError(f"Layer {layer_index} has no submodule `{steer_target}`")
    handle = target.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()



@torch.inference_mode()
def evaluate_with_steering(
    model,
    tok,
    df,
    steer: Optional[torch.Tensor],
    steer_strength: float,
    layer: int,
    steer_target: Literal["residual", "mlp", "self_attn", "post_attn"] = "residual",
    device: str = "cuda",
    *,
    return_preds: bool = False,                
    save_dir: Optional[Path] = None           
):
    """
    Evaluate accuracy **and optionally return / save per-question predictions**.
    
    Returns
    -------
    acc                      : float
    (acc, preds: pd.Series)  : if return_preds is True
    """
    prompts  = df["full_prompt"].tolist()
    answers  = df["correct_letter"].tolist()
    n_choices= df["num_choices"].tolist()
    model.eval()
    preds = []
    correct = 0

    # -------------- steering hook -----------------
    if steer is not None:
        hook_context = apply_steering_hook(model, layer, steer,
                                           steer_strength, steer_target)
    else:
        @contextmanager
        def noop(): yield
        hook_context = noop()
    # ----------------------------------------------

    with hook_context:
        for prompt, true_ans, k in tqdm(
            zip(prompts, answers, n_choices),
            total=len(prompts),
            desc=f"Evaluating at layer {layer} (strength={steer_strength})",
            disable=not sys.stdout.isatty(), leave=False, file=sys.__stdout__
        ):
            valid_letters   = list(ascii_uppercase[:k])
            valid_token_ids = tok.convert_tokens_to_ids(valid_letters)
            if isinstance(valid_token_ids, int):
                valid_token_ids = [valid_token_ids]

            device = next(model.parameters()).device
            enc    = tok(prompt, return_tensors="pt", truncation=True).to(device)
            logits = model(**enc).logits[:, -1, :]
            choice_logits = logits[0, valid_token_ids]
            pred_letter   = valid_letters[torch.argmax(choice_logits).item()]

            preds.append(pred_letter)
            if pred_letter == true_ans:
                correct += 1

    assert not model._forward_hooks, "Hook was not properly removed!"

    acc   = correct / len(answers)
    preds = pd.Series(preds, index=df.index, name="prediction")

    # -------------- optional save -----------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dump = df.copy()
        dump["prediction"] = preds
        dump["is_correct"] = dump["prediction"].eq(dump["correct_letter"])
        dump.to_csv(save_dir / "predictions.csv", index=False)
    # ----------------------------------------------
    return (acc, preds) if return_preds else acc


# ─────────────────────────────── ICL on Wrong-Train Examples ───────────────────────────────
# === small helper to slice wrong rows by a prediction Series ===
def _wrong_rows(df: pd.DataFrame, preds: pd.Series) -> pd.DataFrame:
    return df.loc[preds.ne(df["correct_letter"])]


def _build_icl_prefix(
    tok,
    exemplars: List[Tuple[str, str]],
    test_prompt: str,
    max_length: Optional[int],
) -> str:
    """
    Build an ICL prefix from (full_prompt, correct_letter) exemplars without exceeding the model's max length.
    Keeps *as many exemplars as fit* while preserving the full test prompt.
    """
    if max_length is None or max_length <= 0:
        return "".join([fp + ans + "\n\n" for fp, ans in exemplars])

    # Tokenize the test prompt (must fit intact)
    test_ids = tok(test_prompt, return_tensors="pt", truncation=True).input_ids[0]
    # Conservative buffer (leave a few tokens for safety, e.g., 8)
    safety = 8
    budget = max_length - test_ids.numel() - safety
    if budget <= 0:
        # No room for prefix; return empty prefix
        return ""

    prefix_chunks = []
    used = 0
    for full_prompt, ans in exemplars:
        chunk = f"{full_prompt}{ans}\n\n"
        ids = tok(chunk, return_tensors="pt", truncation=True).input_ids[0]
        if used + ids.numel() > budget:
            break
        prefix_chunks.append(chunk)
        used += ids.numel()

    return "".join(prefix_chunks)

@torch.inference_mode()
def evaluate_with_steering_ICL(
    model,
    tok,
    df_test,
    *,
    wrong_train_df: pd.DataFrame,  
    k: int = 8,
    shuffle: bool = True,
    seed: int = 0,

    # steering (single-vector, same API as evaluate_with_steering)
    steer: Optional[torch.Tensor] = None,
    steer_strength: float = 0.0,
    layer: int = 0,
    steer_target: Literal["residual", "mlp", "self_attn", "post_attn"] = "residual",

    # runtime
    device: str = "cuda",
    return_preds: bool = False,
    save_dir: Optional[Path] = None,
    max_length: Optional[int] = None,  # if None, uses tok.model_max_length when available
):
    """
    Test-time evaluation with In-Context Learning using k examples drawn from
    a precomputed pool of wrong-train items (wrong_train_df).

    wrong_train_df must contain at least columns: ["full_prompt", "correct_letter"].
    """
    # Validate inputs
    for col in ("full_prompt", "correct_letter"):
        if col not in wrong_train_df:
            raise ValueError(f"wrong_train_df must contain column '{col}'.")

    if len(wrong_train_df) == 0:
        # Degenerate case: no wrong examples → fall back to standard evaluation
        return evaluate_with_steering(
            model=model,
            tok=tok,
            df=df_test,
            steer=steer,
            steer_strength=steer_strength,
            layer=layer,
            steer_target=steer_target,
            device=device,
            return_preds=return_preds,
            save_dir=save_dir
        )

    # Resolve tokenizer/model max length if not specified
    if max_length is None:
        max_length = getattr(tok, "model_max_length", None)

    # Pick k exemplars (deterministically if shuffle=False)
    if shuffle:
        chosen = wrong_train_df.sample(
            n=min(k, len(wrong_train_df)),
            random_state=seed
        )
    else:
        chosen = wrong_train_df.iloc[: min(k, len(wrong_train_df))]

    exemplars = list(zip(chosen["full_prompt"].tolist(),
                         chosen["correct_letter"].tolist()))

    # Prepare test data
    prompts   = df_test["full_prompt"].tolist()
    answers   = df_test["correct_letter"].tolist()
    n_choices = df_test["num_choices"].tolist()

    model.eval()
    preds = []
    correct = 0

    # Steering hook
    if steer is not None:
        hook_context = apply_steering_hook(model, layer, steer, steer_strength, steer_target)
    else:
        @contextmanager
        def noop(): yield
        hook_context = noop()

    device = next(model.parameters()).device
    desc = f"Evaluating ICL(k={len(exemplars)}) at layer {layer} (strength={steer_strength})"

    with hook_context:
        for prompt, true_ans, k_choices in tqdm(
            zip(prompts, answers, n_choices),
            total=len(prompts),
            desc=desc,
            disable=not sys.stdout.isatty(),
            leave=False, file=sys.__stdout__
        ):
            # Build safe prefix that preserves the full test prompt
            prefix = _build_icl_prefix(tok, exemplars, prompt, max_length)
            icl_prompt = prefix + prompt

            valid_letters   = list(ascii_uppercase[:k_choices])
            valid_token_ids = tok.convert_tokens_to_ids(valid_letters)
            if isinstance(valid_token_ids, int):
                valid_token_ids = [valid_token_ids]

            enc = tok(icl_prompt, return_tensors="pt", truncation=True).to(device)
            logits = model(**enc).logits[:, -1, :]
            choice_logits = logits[0, valid_token_ids]
            pred_letter   = valid_letters[torch.argmax(choice_logits).item()]

            preds.append(pred_letter)
            if pred_letter == true_ans:
                correct += 1

    assert not model._forward_hooks, "Hook was not properly removed!"

    acc   = correct / len(answers)
    preds = pd.Series(preds, index=df_test.index, name="prediction")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dump = df_test.copy()
        dump["prediction"] = preds
        dump["is_correct"] = dump["prediction"].eq(dump["correct_letter"])
        dump.to_csv(save_dir / "predictions.csv", index=False)
        meta = {
            "icl_k": len(exemplars),
            "icl_indices": getattr(chosen, "index", []).__repr__(),
            "steer_layer": layer,
            "steer_strength": steer_strength,
            "steer_target": steer_target,
            "max_length": max_length,
            "shuffle": shuffle,
            "seed": seed,
        }
        with open(save_dir / "icl_meta.txt", "w") as f:
            for k_, v_ in meta.items():
                f.write(f"{k_}: {v_}\n")

    return (acc, preds) if return_preds else acc




    # ────────────────────── Right→Wrong / Wrong→Right helper ──────────────────────
def compare_predictions(df, unsteered_preds, steered_preds):
    """
    Return two dataframes:
      • R2W – items the baseline got right and the steered model got wrong
      • W2R – items the baseline got wrong and the steered model got right
    """
    unsteered_correct = unsteered_preds.eq(df["correct_letter"])
    steered_correct   = steered_preds.eq(df["correct_letter"])

    r2w_idx = df.index[  unsteered_correct & ~steered_correct]
    w2r_idx = df.index[ ~unsteered_correct &  steered_correct]

    keep_cols = ["question", "correct_letter", "full_prompt"]  \
                    if "question" in df else df.columns
    r2w = df.loc[r2w_idx, keep_cols].copy()
    w2r = df.loc[w2r_idx, keep_cols].copy()

    r2w["unsteered_pred"] = unsteered_preds.loc[r2w_idx]
    r2w["steered_pred"]   = steered_preds.loc[r2w_idx]
    w2r["unsteered_pred"] = unsteered_preds.loc[w2r_idx]
    w2r["steered_pred"]   = steered_preds.loc[w2r_idx]
    return r2w, w2r


# ─────────────────────────────── Open-Ended Evaluation ───────────────────────────────
@torch.inference_mode()
def evaluate_open_ended_with_steering(
    model,
    tok,
    df,
    steer: torch.Tensor | None,
    steer_strength: float,
    layer: int,
    steer_target: str = "residual",
    device: str = "cuda",
    *,
    openai_model: str = "gpt-4o",
    return_preds: bool = False,
    save_dir=None,
):
    """
    Open-ended evaluation with:
      1) steering hook (optional),
      2) model-generated answers,
      3) GPT scorer that grades each answer 0/1 given context+question+true answer.

    Returns
    -------
    test_score : float
    (test_score, preds: pd.Series) if return_preds is True.
    """
    # ---------- Generate answers ----------
    prompts = df["open_ended_prompt"].tolist()   # must exist in df
    true_answers = df["open_ended_true_answer"].tolist()  # must exist in df

    model.eval()
    preds = []

    if steer is not None:
        hook_context = apply_steering_hook(
            model, layer, steer, steer_strength, steer_target
        )
    else:
        @contextmanager
        def noop():
            yield
        hook_context = noop()

    with hook_context:
        for prompt in tqdm(
            prompts,
            total=len(prompts),
            desc=f"Generating open-ended answers (layer={layer}, strength={steer_strength})",
            disable=not sys.stdout.isatty(),
            leave=False,
            file=sys.__stdout__,
        ):
            enc = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(device)


            out = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            generated = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(generated.strip())

    assert not model._forward_hooks, "Hook was not properly removed!"

    preds_series = pd.Series(preds, index=df.index, name="model_answer")

    # ----------Ask GPT to score each (context+question, model answer, true answer) ----------
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    scores: list[float] = []

    for full_prompt, model_ans, true_ans in tqdm(
        list(zip(prompts, preds, true_answers)),
        total=len(prompts),
        desc="Scoring answers with GPT",
        disable=not sys.stdout.isatty(),
        leave=False,
        file=sys.__stdout__,
    ):
        score_instruction = f"""
You are grading a student's short answer.

Context + Question:
-------------------
{full_prompt}

Student's answer:
-----------------
{model_ans}

Reference / true answer:
------------------------
{true_ans}

Instructions:
Return ONLY:
- 1 if the student's answer matches or mostly matches the correct answer.
- 0 if the student's answer is incorrect or incomplete.
- No extra text, output only 0 or 1.
"""

        chat_completion = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "user",
                    "content": score_instruction.strip(),
                }
            ],
        )
        raw_score_text = chat_completion.choices[0].message.content.strip()

        # Extract the first number in the response, just in case
        m = re.search(r"[-+]?\d+(\.\d+)?", raw_score_text)
        if m:
            score_val = float(m.group(0))
        else:
            # Fall back to 0 if GPT returns something unexpected
            score_val = 0.0

        # Clamp to [0,1]
        score_val = max(0.0, min(1.0, score_val))
        scores.append(score_val)

    # ---------- Sanity check lengths ----------
    if len(scores) != len(df):
        raise ValueError(
            f"Length mismatch: got {len(scores)} GPT scores but df has {len(df)} rows."
        )

    scores_series = pd.Series(scores, index=df.index, name="score_0_1")

    test_score = float(np.mean(scores))

    # ---------- Optional save ----------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dump = df.copy()
        dump["model_answer"] = preds_series
        dump["score_0_1"] = scores_series
        dump.to_csv(save_dir / "open_ended_predictions.csv", index=False)

    return (test_score, preds_series) if return_preds else test_score
