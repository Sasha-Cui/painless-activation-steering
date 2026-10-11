# machine_environments.py
from preambles import *
# ─────────────────────────────── Benchmark loading ───────────────────────────────
class BenchmarkLoader:
    """
    Either give absolute split sizes  **or**
    give   (max_sample_size,  train_prop, val_prop, test_prop).
    """
    def __init__(self, *, n_train: int | None = None, n_val:   int | None = None, n_test:  int | None = None, max_sample_size: int | None = None,
        train_prop: float = 0.6, val_prop: float = 0.2, test_prop:  float = 0.2, open_ended: bool = False, seed: int):
        self.raw_n_train = n_train         
        self.raw_n_val   = n_val
        self.raw_n_test  = n_test
        self.max_sample_size = max_sample_size
        self.train_prop = train_prop
        self.val_prop   = val_prop
        self.test_prop  = test_prop
        self.seed = seed
        self.open_ended = open_ended
        # will be filled once we know df size
        self.n_train = self.n_val = self.n_test = None
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load(self):
        raise NotImplementedError

    def _decide_split_sizes(self):
        if self.raw_n_train is not None:      # backwards compatibility path
            self.n_train = self.raw_n_train
            self.n_val   = self.raw_n_val
            self.n_test  = self.raw_n_test
            return

        total = len(self.df)
        if self.max_sample_size is not None:
            total = min(total, self.max_sample_size)

        n_train = int(total * self.train_prop)
        n_val   = int(total * self.val_prop)
        n_test  = int(total * self.test_prop)
        remainder = total - (n_train + n_val + n_test)
        n_train += remainder                  # dump leftovers into train

        self.n_train, self.n_val, self.n_test = n_train, n_val, n_test


    def _split(self):
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        # decide how big each slice is
        self._decide_split_sizes()
        # take only the first `n_train+n_val+n_test` rows if dataset is larger
        slice_end = self.n_train + self.n_val + self.n_test
        self.df = self.df.iloc[:slice_end]

        self.train_df = self.df.iloc[:self.n_train]

        if self.n_val > 0:
            self.val_df = self.df.iloc[self.n_train : self.n_train + self.n_val]
        else:
            self.val_df = self.train_df

        self.test_df = self.df.iloc[self.n_train + self.n_val : self.n_train + self.n_val + self.n_test]

        print(f"✔ Total questions: {len(self.df)}")
        print(f"  • {len(self.train_df)} for training/steering")
        if self.n_val > 0:
            print(f"  • {len(self.val_df)} for validation")
        print(f"  • {len(self.test_df)} for testing")
        if self.open_ended:
            self._make_test_open_ended()


    def _format_mc_question(self, question: str, choices: list, correct_idx: int, context: str = None) -> dict:
        if not (0 <= correct_idx < len(choices)):
            return None

        indexed_choices = list(enumerate(choices))
        rnd = random.Random(self.seed)
        rnd.shuffle(indexed_choices)
        shuffled_choices = [choice for _, choice in indexed_choices]
        shuffled_indices = [i for i, _ in indexed_choices]
        new_correct_index = shuffled_indices.index(correct_idx)

        lettered_choices = {
            letter: choice for letter, choice in zip(string.ascii_uppercase, shuffled_choices)
        }
        correct_letter = string.ascii_uppercase[new_correct_index]

        # Full prompt: includes choices
        mcq_prompt_lines = [question]
        mcq_prompt_lines += [f"{letter}: {text}" for letter, text in lettered_choices.items()]
        if context:
            mcq_prompt_lines.append(f"Remember, {context}.")
        mcq_prompt_lines.append("Respond with a single letter.\nAnswer:")
        full_prompt = "\n".join(mcq_prompt_lines)

        return {
            "question": question,                           # string
            "choices": lettered_choices,                    # dict: A → "Red", B → "Blue", etc.
            "shuffled_choices": shuffled_choices,           # list of strings
            "correct_letter": correct_letter,               # e.g., "C"
            "correct_choice": lettered_choices[correct_letter],  # e.g., "Yellow"
            "full_prompt": full_prompt,                     # rendered string
            "num_choices": len(lettered_choices)
        }

    def _make_open_ended_prompt(self, row: pd.Series) -> str:
        base = str(row.get("question", "")).rstrip()
        if not base:
            base = ""  # fallback so we don't crash
        return base + "\nAnswer:"
    def _make_test_open_ended(self):
        """
        Convert the test split to open-ended:
          - keep all MC metadata (choices, correct_letter, etc.)
          - create 'open_ended_prompt'
          - overwrite 'full_prompt' so downstream code sees open-ended prompts
        """
        if self.test_df is None or len(self.test_df) == 0:
            return

        df = self.test_df.copy()
        # keep the original MC full prompt for reference
        if "full_prompt" in df.columns:
            df["full_prompt_mc"] = df["full_prompt"]

        df["open_ended_prompt"] = df.apply(self._make_open_ended_prompt, axis=1)

        # build ground-truth open-ended answer
        if "correct_choice" in df.columns:
            df["open_ended_true_answer"] = df["correct_choice"]
        elif "correct_letter" in df.columns:
            df["open_ended_true_answer"] = df["correct_letter"]
        else:
            raise KeyError(
                "Cannot construct 'open_ended_true_answer': neither "
                "'correct_choice' nor 'correct_letter' present in test_df."
            )

        # Make the model actually see the open-ended version at test time
        df["full_prompt"] = df["open_ended_prompt"]

        self.test_df = df

    def get(self):
        return self.train_df.copy(), self.val_df.copy(), self.test_df.copy()

    def split_meta(self) -> dict:
        return dict(n_train=self.n_train, n_val=self.n_val, n_test=self.n_test)


# Helper functions for loading the benchmarks
def strip_choice_prefix(choice: str) -> str:
    return re.sub(r"^\([A-E]\)\s*", "", choice).strip()

def remove_inline_choices_from_query(query: str) -> str:
    # Remove embedded "Answer Choices: ..." block
    query = re.sub(r'Answer Choices:.*?(?=\n|$)', '', query, flags=re.IGNORECASE)

    # Remove "Q:" style prefixes mid-text
    query = re.sub(r'\bQ\s*[:\.]\s*', '', query)
    query = re.sub(r'([a-z])Q[:\.]', r'\1. ', query)

    # Remove trailing prompt lines like "A: Among A through E, the answer is"
    query = re.sub(r'A:\s*Among A through E, the answer is.*', '', query, flags=re.IGNORECASE)

    return query.strip()


# Benchmark Loading function
def load_benchmark(benchmark_name: str, *, max_sample_size: int | None = None, train_prop: float = 0.6, val_prop:   float = 0.2, test_prop:  float = 0.2, open_ended: bool = False, verbose_on = False, seed: int):
    if benchmark_name not in benchmark_map:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    benchmark_class = benchmark_map[benchmark_name]
    benchmark = benchmark_class(max_sample_size = max_sample_size, train_prop  = train_prop, val_prop = val_prop, test_prop = test_prop, open_ended=open_ended, seed=seed)
    benchmark.load()

    train_df, val_df, test_df = benchmark.get()
    if verbose_on:
        print("📎 A benchmark full_prompt example:")
        print("Question:", train_df.iloc[10]['full_prompt'],"\n")
        print("Correct answer:", train_df.iloc[10]['correct_letter'], "\n\n")

    return train_df, val_df, test_df, benchmark.split_meta()




class HighSchoolMathMMLU(BenchmarkLoader):
    """
    High-school math subset of MMLU.

    Parameters
    ----------
    subject : str
        Which MMLU subject to keep (default "high_school_mathematics").
    **kwargs : forwarded to BenchmarkLoader
        Either give (n_train, n_val, n_test) **or**
        (max_sample_size, train_prop, val_prop, test_prop).
    """
    def __init__(self, subject: str = "high_school_mathematics", **kwargs):
        super().__init__(**kwargs)
        self.subject = subject

    def load(self):
        """Load the chosen MMLU subject and prepare the MCQ dataframe."""
        ds = load_dataset(
            "cais/mmlu",
            self.subject,
            split="test",
            trust_remote_code=True,
        )

        rows = []
        for row in ds:
            question = row.get("question")
            choices = row.get("choices")
            correct_idx = row.get("answer")
            formatted = self._format_mc_question(question, choices, correct_idx)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print("\n\n>>> Loaded High School Mathematics MC benchmark from MMLU.")
        self._split()
        
from datasets import load_dataset, get_dataset_config_names

class MMLU(BenchmarkLoader):
    """
    Load all MMLU subjects except 'all' and 'auxiliary_train'.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exclude = {"all", "auxiliary_train"}

    def load(self):
        # List available subject configs
        subjects = get_dataset_config_names("cais/mmlu")
        subjects_to_load = [s for s in subjects if s not in self._exclude]

        rows = []
        for subj in subjects_to_load:
            ds = load_dataset(
                "cais/mmlu",
                subj,
                split="test",
                trust_remote_code=True,
            )
            for row in ds:
                question = row.get("question")
                choices = row.get("choices")
                correct_idx = row.get("answer")
                formatted = self._format_mc_question(question, choices, correct_idx)
                if formatted:
                    formatted["subject"] = subj  # optional, useful for analysis
                    rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print(f"\n\n>>> Loaded MMLU subjects (n={len(subjects_to_load)}), excluded {self._exclude}.")
        self._split()



class TruthfulQA(BenchmarkLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def load(self):
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", trust_remote_code=True, revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
        rows = []

        for question, target in zip(ds["question"], ds["mc1_targets"]):
            choices = target["choices"]
            labels = target["labels"]
            try:
                correct_idx = labels.index(1)
            except ValueError:
                continue

            formatted = self._format_mc_question(question, choices, correct_idx)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print("\n\n>>> Loaded TruthfulQA benchmark data from validation split")
        self._split()


class OpenBookQA(BenchmarkLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def load(self):
        ds = load_dataset("allenai/openbookqa", "additional", split="train", trust_remote_code=True, revision="388097ea7776314e93a529163e0fea805b8a6454")
        rows = []

        for example in ds:
            question_stem = example["question_stem"]
            fact = example["fact1"]
            full_question = f"{question_stem}"

            choices = example["choices"]["text"]  # list of choices
            labels = example["choices"]["label"]  # list like ['A', 'B', 'C', 'D']
            answer_key = example["answerKey"]      # correct letter: 'A', 'B', etc.

            if answer_key not in labels:
                continue  # skip malformed

            correct_idx = labels.index(answer_key)

            formatted = self._format_mc_question(full_question, choices, correct_idx, fact)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print("\n\n>>> Loaded OpenBookQA benchmark (train split with context fact1).")
        self._split()




class CommonsenseQA(BenchmarkLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def load(self):
        ds = load_dataset("tau/commonsense_qa", split="train", trust_remote_code=True, revision="94630fe30dad47192a8546eb75f094926d47e155")
        rows = []

        for example in ds:
            question = example["question"]
            choices = example["choices"]["text"]  # list of answer options
            answer_letter = example["answerKey"]  # e.g., "B"

            if answer_letter not in string.ascii_uppercase[:len(choices)]:
                continue  # skip malformed entries

            correct_idx = string.ascii_uppercase.index(answer_letter)

            formatted = self._format_mc_question(question, choices, correct_idx)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print("\n\n>>> Loaded CommonsenseQA benchmark (train split).")
        self._split()

class ARCChallenge(BenchmarkLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def load(self):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True, revision="210d026faf9955653af8916fad021475a3f00453")
        rows = []

        for example in ds:
            question = example["question"]
            choices = example["choices"]["text"]  # list of answer options
            answer_letter = example["answerKey"]  # e.g., "A", "B", "C", "D"

            if answer_letter not in string.ascii_uppercase[:len(choices)]:
                continue  # skip malformed entries

            correct_idx = string.ascii_uppercase.index(answer_letter)

            formatted = self._format_mc_question(question, choices, correct_idx)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print("\n\n>>> Loaded AI2 ARC-Challenge benchmark (train split).")
        self._split()





class LSAT(BenchmarkLoader):
    """Loader for the LSAT (Law School Admission Test) benchmark.

    This class loads the full LSAT dataset or specific sections (LR, RC, AR).
    Each item includes:
        - query: str (the question text)
        - choices: list of str (the answer choices)
        - gold: list of int (index of the correct answer)
    """
    def __init__(self, subsets: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        default = ["lr", "rc", "ar"]
        self.subsets = subsets or default

    def load(self):
        rows: list[dict] = []
        dataset_map = {
            "lr": ("hails/agieval-lsat-lr", "d876c675a8d47aa4d8a6d682ca8400b7d2ffe1c4"),
            "rc": ("hails/agieval-lsat-rc", "432868ea4fa7b50db66d14524d42472dd052b53c"),
            "ar": ("hails/agieval-lsat-ar", "052cc636b612f5563329dd182fb6c2cad56681c8"),
        }

        datasets_to_concatenate = []
        for subset in self.subsets:
            if subset not in dataset_map:
                print(f"⚠️  Skipping LSAT subset '{subset}' – unknown subset.")
                continue
            try:
                path, revision = dataset_map[subset]
                ds = load_dataset(path, split="test", trust_remote_code=True, revision=revision)
                datasets_to_concatenate.append(ds)
            except Exception as e:
                print(f"⚠️  Skipping LSAT subset '{subset}' – load failed ({e}).")
                continue

        if not datasets_to_concatenate:
            print("No LSAT subsets loaded.")
            self.df = pd.DataFrame()
            return

        ds = concatenate_datasets(datasets_to_concatenate)

        for example in ds:
            raw_question = example["query"]
            raw_choices = example["choices"]  # includes (A)...(E)
            gold = example["gold"]            # e.g., [2]

            if not isinstance(gold, list) or len(gold) != 1:
                continue

            correct_idx = gold[0]
            if not (0 <= correct_idx < len(raw_choices)):
                continue

            # Clean up: remove embedded choices and prefix labels
            question = remove_inline_choices_from_query(raw_question)
            choices = [strip_choice_prefix(choice) for choice in raw_choices]

            formatted = self._format_mc_question(question, choices, correct_idx)
            if formatted:
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print(f"\n\n>>> Loaded LSAT benchmark ({', '.join(self.subsets)} splits).")
        self._split()

class LSAT_LR(LSAT):
    def __init__(self, **kwargs):
        super().__init__(subsets=["lr"], **kwargs)

class LSAT_RC(LSAT):
    def __init__(self, **kwargs):
        super().__init__(subsets=["rc"], **kwargs)

class LSAT_AR(LSAT):
    def __init__(self, **kwargs):
        super().__init__(subsets=["ar"], **kwargs)


# ─────────────────────────────── Moody Benchmarks ───────────────────────────────
class BBQ(BenchmarkLoader):
    """Loader for the BBQ (Bias Benchmark for QA) multiple-choice benchmark.

    This class loads all or a specified subset of the 11 social bias categories:
        - age, disability_status, gender_identity, nationality, physical_appearance,
          race_ethnicity, race_x_gender, race_x_ses, religion, ses, sexual_orientation

    Each item includes:
        - context: str
        - question: str
        - ans0, ans1, ans2: answer choices
        - answer_label: index of correct answer (0–2)
    """
    def __init__(self, subsets: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        default = [
            "age",
            "disability_status",
            "gender_identity",
            "nationality",
            "physical_appearance",
            "race_ethnicity",
            "race_x_gender",
            "race_x_ses",
            "religion",
            "ses",
            "sexual_orientation",
        ]
        self.subsets = subsets or default

    def load(self):
        rows: list[dict] = []

        for subset in self.subsets:
            try:
                ds = load_dataset("Elfsong/BBQ", "default", split=subset, trust_remote_code=True, revision="56cd2dcb60a10288b4f1a9c8cff6cf120141c08a")
            except Exception as e:
                print(f"⚠️  Skipping BBQ subset '{subset}' – load failed ({e}).")
                continue

            for ex in ds:
                context = ex.get("context")
                question = ex.get("question")
                choices = [ex.get("ans0"), ex.get("ans1"), ex.get("ans2")]
                correct_idx = int(ex.get("answer_label", -1))

                if (context is None) or (question is None) or (None in choices) or not (0 <= correct_idx < len(choices)):
                    continue  # skip malformed

                full_question = f"Context: {context}\nQuestion: {question}"
                formatted = self._format_mc_question(full_question, choices, correct_idx)
                if formatted:
                    formatted["subset"] = subset
                    rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print(f"\n\n>>> Loaded BBQ benchmark – {len(self.df)} questions from subsets: {', '.join(self.subsets)}.")
        self._split()

class Age(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["age"], **kwargs)

class DisabilityStatus(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["disability_status"], **kwargs)

class GenderIdentity(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["gender_identity"], **kwargs)

class Nationality(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["nationality"], **kwargs)

class PhysicalAppearance(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["physical_appearance"], **kwargs)

class RaceEthnicity(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["race_ethnicity"], **kwargs)

class RaceXGender(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["race_x_gender"], **kwargs)

class RaceXSES(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["race_x_ses"], **kwargs)

class Religion(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["religion"], **kwargs)

class SES(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["ses"], **kwargs)

class SexualOrientation(BBQ):
    def __init__(self, **kwargs):
        super().__init__(subsets=["sexual_orientation"], **kwargs)


class ETHICS(BenchmarkLoader):
    """Loader for Hendrycks ETHICS-style multiple-choice benchmark.

    Each example includes:
        - scenario: str
        - excuse: Optional[str]
        - label: int (0 = acceptable, 1 = unacceptable)

    All questions are phrased as: "Is this morally acceptable?"
    with choices:
        A: Yes, it is morally acceptable.
        B: No, it is morally unacceptable.
    """
    def __init__(self, subsets: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        default = [
            "commonsense",
            "deontology",
            "justice",
        ]
        self.subsets = subsets or default

    def load(self):
        rows: list[dict] = []

        for subset in self.subsets:
            try:
                ds = load_dataset("hendrycks/ethics", subset, split="train+validation+test", trust_remote_code=True, revision="b8b47c589f8bee77175b8648e5497278b68da48a")
            except Exception as e:
                print(f"⚠️  Skipping ETHICS subset '{subset}' – load failed ({e}).")
                continue

            for ex in ds:
                scenario = ex.get("scenario") or ex.get("input")
                excuse = ex.get("excuse", "").strip()
                label = ex.get("label")

                if (scenario is None) or (label not in [0, 1]):
                    continue  # skip malformed

                # Build the MCQ prompt
                if excuse:
                    full_prompt = f"Scenario: {scenario}\nResponse: {excuse}\n\nIs the above an acceptable way to respond to the question?"
                else:
                    full_prompt = f"Scenario: {scenario}\n\nIs this morally acceptable?"

                choices = [
                    "Yes, it is acceptable.",     # index 0
                    "No, it is unacceptable.",     # index 1
                ]
                correct_idx = label if subset == "commonsense" else int(not label) # bad practice in benchmark making

                formatted = self._format_mc_question(full_prompt, choices, correct_idx)
                if formatted:
                    rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print(f"\n\n>>> Loaded ETHICS benchmark – {len(self.df)} questions from subsets: {', '.join(self.subsets)}.")
        self._split()


class ETHICS_Commonsense(ETHICS):
    def __init__(self, **kwargs):
        super().__init__(subsets=["commonsense"], **kwargs)

class ETHICS_Deontology(ETHICS):
    def __init__(self, **kwargs):
        super().__init__(subsets=["deontology"], **kwargs)

class ETHICS_Justice(ETHICS):
    def __init__(self, **kwargs):
        super().__init__(subsets=["justice"], **kwargs)


class Sycophancy(BenchmarkLoader):
    """
    EleutherAI Sycophancy: three sub-datasets of model-written evaluations of sycophantic behavior.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.configs = [
            "sycophancy_on_nlp_survey",
            "sycophancy_on_philpapers2020",
            "sycophancy_on_political_typology_quiz",
        ]

    def load(self):
        rows = []
        for cfg in self.configs:
            ds = load_dataset("EleutherAI/sycophancy", cfg, split="validation", trust_remote_code=True, revision="a2a807741737a640a5556bc9ff4049494fabae73")
            for ex in ds:
                question = ex["question"]
                # two choices: the sycophantic behavior vs. the non-sycophantic
                choices = [
                    ex["answer_matching_behavior"],
                    ex["answer_not_matching_behavior"],
                ]
                correct_idx = 0  # matching_behavior is the "positive" label
                formatted = self._format_mc_question(question, choices, correct_idx)
                if not formatted:
                    continue
                if cfg == "sycophancy_on_political_typology_quiz":
                    aff = ex.get("user_affiliation")
                    formatted["full_prompt"] = (
                        formatted["full_prompt"]
                        + f"\n(Note: user affiliation: {aff})"
                    )
                rows.append(formatted)

        self.df = pd.DataFrame(rows)
        print(f"\n\n>>> Loaded Sycophancy (configs: {', '.join(self.configs)}) – {len(self.df)} examples")
        self._split()





benchmark_map = {
    "HighSchoolMathMMLU": HighSchoolMathMMLU,
    "MMLU": MMLU,
    "OpenBookQA": OpenBookQA,
    "CommonsenseQA": CommonsenseQA,
    "ARCChallenge": ARCChallenge,
    "LSAT": LSAT,
    "TruthfulQA": TruthfulQA,
    "BBQ": BBQ,
    "ETHICS": ETHICS,
    "ETHICS_Commonsense": ETHICS_Commonsense,
    "ETHICS_Deontology": ETHICS_Deontology,
    "ETHICS_Justice": ETHICS_Justice,
    "Sycophancy": Sycophancy,
    "DisabilityStatus": DisabilityStatus,
    "GenderIdentity": GenderIdentity,
    "Nationality": Nationality,
    "PhysicalAppearance": PhysicalAppearance,
    "RaceEthnicity": RaceEthnicity,
    "RaceXGender": RaceXGender,
    "RaceXSES": RaceXSES,
    "Religion": Religion,
    "SES": SES,
    "SexualOrientation": SexualOrientation,
    "LSAT_LR": LSAT_LR,
    "LSAT_RC": LSAT_RC,
    "LSAT_AR": LSAT_AR,
}
