from __future__ import annotations

import json
import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime in notebook envs
    np = None

# Shared regexes for GSM8K-style answers and numeric extraction.
final_re = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

_JUDGE_TOKENIZER = None
_JUDGE_MODEL = None


def _require_numpy() -> None:
    if np is None:
        raise ImportError("numpy is required for ensemble/math utilities in hw2_helpers.py")


def show_example(example: Dict[str, str]) -> None:
    print("\nQUESTION:\n", example["question"])
    print("\nSOLUTION:\n", example["answer"])
    if "true_answer" in example:
        print("\nTRUE ANSWER:", example["true_answer"])


def extract_true_answer(example: Dict[str, str]) -> Optional[int]:
    match = final_re.search(example["answer"])
    if match:
        return int(float(match.group(1)))
    return None


def extract_candidate_answer(candidate_text: str) -> Optional[int]:
    """
    Extract the final numeric answer from GSM8K-style text.
    Returns an int if found, otherwise None.
    """
    match = final_re.search(candidate_text)
    if match:
        return int(float(match.group(1)))
    return None


def verifier_exact_match(candidate_text: str, true_answer: Optional[int]) -> float:
    predicted_answer = extract_candidate_answer(candidate_text)
    if predicted_answer is None or true_answer is None:
        return 0.0
    return 1.0 if predicted_answer == int(true_answer) else 0.0


def verifier_numeric_distance(candidate_text: str, true_answer: Optional[int]) -> float:
    predicted_answer = extract_candidate_answer(candidate_text)
    if predicted_answer is None or true_answer is None:
        return 0.0

    distance = abs(predicted_answer - int(true_answer))
    return 1.0 / (1.0 + distance)


def verifier_format_confidence(candidate_text: str) -> float:
    """
    Heuristic confidence that text has one clean final GSM8K answer line.
    """
    matches = final_re.findall(candidate_text)
    if len(matches) == 0:
        return 0.0
    if len(matches) > 1:
        return 0.4
    return 1.0 if "####" in candidate_text[-60:] else 0.8


def verifier_step_consistency(candidate_text: str) -> float:
    """
    Heuristic process verifier for obvious reasoning inconsistencies.
    """
    text_lower = candidate_text.lower()

    if ("division by zero" in text_lower) or ("/ 0" in text_lower) or ("÷ 0" in text_lower):
        return 0.0

    unit_words = (
        r"(apples|oranges|dollars|people|students|cars|items|cookies|"
        r"marbles|books|pencils|minutes|hours)"
    )
    neg_count = re.search(rf"\b-+\d+\s+{unit_words}\b", text_lower) is not None

    eq_lines = [line.strip() for line in candidate_text.splitlines() if "=" in line]
    checked = 0
    ok = 0

    for line in eq_lines:
        line = line.replace("×", "*")
        match = re.search(
            r"([-+]?\d+)\s*=\s*([-+]?\d+)\s*([\+\-\*\/])\s*([-+]?\d+)",
            line,
        )
        if not match:
            continue

        a = int(match.group(1))
        b = int(match.group(2))
        op = match.group(3)
        c = int(match.group(4))
        checked += 1

        try:
            if op == "+" and a == b + c:
                ok += 1
            elif op == "-" and a == b - c:
                ok += 1
            elif op == "*" and a == b * c:
                ok += 1
            elif op == "/" and c != 0 and a == b / c:
                ok += 1
        except Exception:
            pass

    score = 1.0
    if neg_count:
        score -= 0.3
    if checked > 0:
        score *= ok / checked

    return max(0.0, min(1.0, score))


def load_judge_model(model_name: str):
    """Load and return tokenizer/model pair for LLM step judging."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    judge_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    return judge_tokenizer, judge_model


def set_judge_components(judge_tokenizer, judge_model) -> None:
    """Set global tokenizer/model used by judge helper functions."""
    global _JUDGE_TOKENIZER, _JUDGE_MODEL
    _JUDGE_TOKENIZER = judge_tokenizer
    _JUDGE_MODEL = judge_model


def split_into_steps(candidate_text: str, max_steps: int = 12) -> List[str]:
    text = re.sub(r"####.*", "", candidate_text).strip()
    if not text:
        return []

    if "\n" in text:
        steps = [line.strip() for line in text.split("\n") if line.strip()]
    else:
        steps = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]

    return steps[:max_steps]


def build_step_judge_prompt(question: str, prev_steps: Sequence[str], current_step: str) -> str:
    prev = "\n".join([f"{i+1}. {s}" for i, s in enumerate(prev_steps)]) if prev_steps else "(none yet)"

    return f"""
You are a strict math reasoning step checker.

Decide whether the CURRENT step is valid given:
- the QUESTION
- the PREVIOUS steps

Return ONLY valid JSON with keys:
- "verdict": "valid" or "invalid"
- "confidence": a number from 0 to 1
- "reason": a short explanation (max 20 words)

QUESTION:
{question}

PREVIOUS STEPS:
{prev}

CURRENT STEP:
{current_step}
""".strip()


def parse_judge_json(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"verdict": "invalid", "confidence": 0.0, "reason": "no json found"}

    try:
        obj = json.loads(match.group(0))
        verdict = str(obj.get("verdict", "")).lower().strip()
        confidence = float(obj.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        if verdict not in ("valid", "invalid"):
            verdict = "invalid"
            confidence = 0.0

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": str(obj.get("reason", ""))[:200],
        }
    except Exception:
        return {"verdict": "invalid", "confidence": 0.0, "reason": "json parse error"}


def judge_step_validity(
    question: str,
    prev_steps: Sequence[str],
    current_step: str,
    max_new_tokens: int = 120,
    temperature: float = 0.0,
    judge_tokenizer=None,
    judge_model=None,
) -> Tuple[float, Dict[str, object], str]:
    """
    Judge one step and return (score, parsed_json, raw_generation_text).
    """
    tokenizer = judge_tokenizer or _JUDGE_TOKENIZER
    model = judge_model or _JUDGE_MODEL

    if tokenizer is None or model is None:
        parsed = {"verdict": "invalid", "confidence": 0.0, "reason": "judge model not loaded"}
        return 0.0, parsed, ""

    prompt = build_step_judge_prompt(question, prev_steps, current_step)

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    import torch

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 0.001),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parsed = parse_judge_json(raw)
    score = parsed["confidence"] if parsed["verdict"] == "valid" else 0.0

    return float(score), parsed, raw


def verifier_step_plausibility_judge(
    candidate_text: str,
    question: str,
    max_steps: int = 8,
    judge_tokenizer=None,
    judge_model=None,
) -> float:
    steps = split_into_steps(candidate_text, max_steps=max_steps)
    if len(steps) == 0:
        return 0.0

    step_scores = []
    prev_steps = []
    for step in steps:
        score, _, _ = judge_step_validity(
            question,
            prev_steps,
            step,
            judge_tokenizer=judge_tokenizer,
            judge_model=judge_model,
        )
        step_scores.append(score)
        prev_steps.append(step)

    return float(sum(step_scores) / len(step_scores))


def extract_all_numbers(text: str) -> List[str]:
    return number_re.findall(text)


def split_reasoning_and_final(candidate_text: str) -> Tuple[str, str]:
    match = re.search(r"(####\s*[-+]?\d+(?:\.\d+)?)", candidate_text)
    if not match:
        return candidate_text, ""
    idx = match.start()
    return candidate_text[:idx].strip(), candidate_text[idx:].strip()


def verifier_step_to_outcome_coherence(
    candidate_text: str,
    true_answer: Optional[int] = None,
    max_steps: int = 12,
) -> float:
    final_pred = extract_candidate_answer(candidate_text)
    if final_pred is None:
        return 0.0

    reasoning_text, _ = split_reasoning_and_final(candidate_text)
    steps = split_into_steps(candidate_text, max_steps=max_steps)

    if len(steps) == 0:
        return 0.2

    reasoning_numbers = extract_all_numbers(reasoning_text)
    step_numbers = [extract_all_numbers(step) for step in steps]

    final_str = str(final_pred)
    final_mentioned_early = final_str in reasoning_numbers

    steps_with_numbers = sum(1 for nums in step_numbers if len(nums) > 0)
    frac_steps_numeric = steps_with_numbers / len(steps)

    flat_numbers = [n for nums in step_numbers for n in nums]
    if len(flat_numbers) == 0:
        continuity = 0.0
    else:
        counts = {}
        for n in flat_numbers:
            counts[n] = counts.get(n, 0) + 1
        repeated = sum(1 for _, count in counts.items() if count >= 2)
        continuity = repeated / max(1, len(counts))

    suspicious_jump = (not final_mentioned_early) and (len(reasoning_numbers) <= 2)

    score = 0.35
    score += 0.30 if final_mentioned_early else 0.0
    score += 0.20 * frac_steps_numeric
    score += 0.15 * continuity

    if suspicious_jump:
        score -= 0.25

    if true_answer is not None:
        try:
            if int(final_pred) == int(true_answer):
                score += 0.05
        except Exception:
            pass

    return max(0.0, min(1.0, score))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def zscore_sigmoid(
    x: float,
    mu: float,
    sigma: float,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> float:
    sigma = max(float(sigma), eps)
    z = (x - mu) / sigma
    return float(sigmoid(z / max(float(temperature), eps)))


def rank_normalize(scores: Sequence[float]) -> List[float]:
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    sorted_idx = sorted(range(n), key=lambda i: scores[i])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[sorted_idx[j + 1]] == scores[sorted_idx[i]]:
            j += 1

        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank

        i = j + 1

    return [float(r / (n - 1)) for r in ranks]


def score_candidates_raw(
    question: str,
    true_answer: Optional[int],
    candidate_texts: Sequence[str],
    use_llm_judge: bool = True,
    judge_tokenizer=None,
    judge_model=None,
) -> Dict[str, List[float]]:
    raw = {
        "exact_match": [],
        "numeric_distance": [],
        "format_conf": [],
        "step_consistency": [],
        "step_plausibility": [],
        "step_coherence": [],
    }

    for candidate in candidate_texts:
        raw["exact_match"].append(verifier_exact_match(candidate, true_answer))
        raw["numeric_distance"].append(verifier_numeric_distance(candidate, true_answer))
        raw["format_conf"].append(verifier_format_confidence(candidate))
        raw["step_consistency"].append(verifier_step_consistency(candidate))

        if use_llm_judge:
            step_plaus = verifier_step_plausibility_judge(
                candidate,
                question,
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
            )
        else:
            step_plaus = 0.0

        raw["step_plausibility"].append(step_plaus)
        raw["step_coherence"].append(
            verifier_step_to_outcome_coherence(candidate, true_answer=None)
        )

    return raw


def make_demo_candidates_from_gold(example: Dict[str, object]) -> List[str]:
    gold = example["answer"]
    true_answer = int(example["true_answer"])

    candidates = [gold]
    candidates.append(re.sub(r"####.*", "I think the answer is 12.", gold))
    candidates.append(f"Some reasoning...\n#### {true_answer + 7}")
    candidates.append("\n".join(gold.splitlines()[:3]) + f"\n#### {true_answer}")

    return candidates


def compute_global_stats(
    heldout_dataset,
    num_examples: int = 40,
    use_llm_judge: bool = True,
    judge_tokenizer=None,
    judge_model=None,
) -> Dict[str, Dict[str, float]]:
    all_scores = {
        "exact_match": [],
        "numeric_distance": [],
        "format_conf": [],
        "step_consistency": [],
        "step_plausibility": [],
        "step_coherence": [],
    }

    for i in range(min(num_examples, len(heldout_dataset))):
        example = heldout_dataset[i]
        question = example["question"]
        true_answer = example["true_answer"]
        candidates = make_demo_candidates_from_gold(example)

        raw_scores = score_candidates_raw(
            question,
            true_answer,
            candidates,
            use_llm_judge=use_llm_judge,
            judge_tokenizer=judge_tokenizer,
            judge_model=judge_model,
        )
        for key in all_scores.keys():
            all_scores[key].extend(raw_scores[key])

    stats = {}
    for key, values in all_scores.items():
        vals = [float(v) for v in values]
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        else:
            mean = 0.0
            std = 1.0
        stats[key] = {
            "mean": float(mean),
            "std": float(std),
        }

    return stats


def normalize_scores(
    raw_scores: Dict[str, Sequence[float]],
    global_stats: Dict[str, Dict[str, float]],
    temperature: float = 1.0,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    norm_zsig: Dict[str, List[float]] = {}
    norm_rank: Dict[str, List[float]] = {}

    for verifier_name, scores in raw_scores.items():
        mu = global_stats[verifier_name]["mean"]
        sd = global_stats[verifier_name]["std"]

        norm_zsig[verifier_name] = [
            zscore_sigmoid(s, mu, sd, temperature=temperature) for s in scores
        ]
        norm_rank[verifier_name] = rank_normalize(scores)

    return norm_zsig, norm_rank


def _replace_final_answer(candidate_text: str, new_value: int) -> str:
    if final_re.search(candidate_text):
        return final_re.sub(f"#### {int(new_value)}", candidate_text)
    return candidate_text.strip() + f"\n#### {int(new_value)}"


def make_deceptive_wrong(candidate_text: str, nudge: int = 1) -> str:
    """
    Corrupt one arithmetic statement while preserving fluent structure.
    Falls back to nudging the final answer.
    """
    lines = candidate_text.splitlines()
    add_pat = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")

    for i, line in enumerate(lines):
        match = add_pat.search(line)
        if match:
            a, b, c = map(int, match.groups())
            lines[i] = add_pat.sub(f"{a} + {b} = {c + int(nudge)}", line)
            return "\n".join(lines)

    pred = extract_candidate_answer(candidate_text)
    if pred is not None:
        return _replace_final_answer(candidate_text, pred + int(nudge))

    return candidate_text.strip() + f"\n#### {int(nudge)}"


def make_fixed_candidate_pool_from_gold(
    example: Dict[str, object],
    k: int = 8,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """
    Build a deterministic fixed candidate set from one GSM8K example.
    This stays generation-free and creates diverse weak-verifier failure modes.
    """
    rng = random.Random(seed)
    gold = str(example["answer"])
    true_answer = int(example["true_answer"])
    question = str(example["question"])

    variants: List[str] = []
    variants.append(gold)  # gold
    variants.append(_replace_final_answer(gold, true_answer + 1))
    variants.append(_replace_final_answer(gold, true_answer - 1))
    variants.append(_replace_final_answer(gold, true_answer + 7))
    variants.append(re.sub(r"####\s*[-+]?\d+(?:\.\d+)?", "", gold).strip())
    variants.append(
        "\n".join(split_into_steps(gold)[: max(1, min(3, len(split_into_steps(gold))))])
        + f"\n#### {true_answer}"
    )
    variants.append(
        "We compute: 12 = 5 + 6\nThis looks consistent at a glance.\n"
        + f"Therefore, #### {true_answer + 3}"
    )
    variants.append(
        f"Let's reason carefully about the quantities in the question.\n"
        f"The structure is correct, but we make a subtle arithmetic slip.\n#### {true_answer + 2}"
    )
    variants.append(make_deceptive_wrong(gold, nudge=rng.choice([1, 2])))

    while len(variants) < int(k):
        delta = rng.choice([-9, -5, -3, 3, 5, 9])
        variants.append(_replace_final_answer(gold, true_answer + delta))

    unique_texts: List[str] = []
    seen = set()
    for text in variants:
        key = text.strip()
        if key not in seen:
            unique_texts.append(key)
            seen.add(key)
        if len(unique_texts) >= int(k):
            break

    records: List[Dict[str, object]] = []
    for cand_text in unique_texts:
        final = extract_candidate_answer(cand_text)
        records.append(
            {
                "text": cand_text,
                "final": "" if final is None else f"#### {final}",
                "steps": split_into_steps(cand_text),
                "question_hint": question[:80],
            }
        )
    return records


def build_fixed_candidate_pool_records(
    dataset_split,
    k: int = 8,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    n = len(dataset_split) if max_examples is None else min(len(dataset_split), int(max_examples))

    for idx in range(n):
        ex = dataset_split[idx]
        rec = {
            "id": f"gsm8k_{idx}",
            "question": ex["question"],
            "true_answer": int(ex["true_answer"]),
            "candidates": make_fixed_candidate_pool_from_gold(ex, k=k, seed=seed + idx),
        }
        records.append(rec)

    return records


def save_jsonl_records(records: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_jsonl_records(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def score_problem_record_raw(
    problem_record: Dict[str, object],
    use_llm_judge: bool = True,
    judge_tokenizer=None,
    judge_model=None,
) -> Dict[str, List[float]]:
    candidate_texts = [cand["text"] for cand in problem_record["candidates"]]
    return score_candidates_raw(
        question=str(problem_record["question"]),
        true_answer=int(problem_record["true_answer"]),
        candidate_texts=candidate_texts,
        use_llm_judge=use_llm_judge,
        judge_tokenizer=judge_tokenizer,
        judge_model=judge_model,
    )


def compute_global_stats_from_pool(
    pool_records: Sequence[Dict[str, object]],
    max_examples: Optional[int] = None,
    use_llm_judge: bool = True,
    judge_tokenizer=None,
    judge_model=None,
) -> Dict[str, Dict[str, float]]:
    all_scores = {
        "exact_match": [],
        "numeric_distance": [],
        "format_conf": [],
        "step_consistency": [],
        "step_plausibility": [],
        "step_coherence": [],
    }

    n = len(pool_records) if max_examples is None else min(len(pool_records), int(max_examples))
    for i in range(n):
        raw = score_problem_record_raw(
            pool_records[i],
            use_llm_judge=use_llm_judge,
            judge_tokenizer=judge_tokenizer,
            judge_model=judge_model,
        )
        for key in all_scores:
            all_scores[key].extend(raw[key])

    stats: Dict[str, Dict[str, float]] = {}
    for key, values in all_scores.items():
        vals = [float(v) for v in values]
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        else:
            mean = 0.0
            std = 1.0
        stats[key] = {"mean": float(mean), "std": float(std)}

    return stats


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    _require_numpy()
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    if np.std(arr_a) < 1e-8 or np.std(arr_b) < 1e-8:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    _require_numpy()
    x = np.array(list(values), dtype=float) / max(float(temperature), 1e-8)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)


def dict_to_matrix(
    norm_scores_dict: Dict[str, Sequence[float]],
    verifier_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    _require_numpy()
    names = list(norm_scores_dict.keys()) if verifier_names is None else list(verifier_names)
    mat = np.stack([np.array(norm_scores_dict[v], dtype=float) for v in names], axis=0)
    return names, mat


def loo_agreement_weights(
    norm_scores_dict: Dict[str, Sequence[float]],
    temperature: float = 1.0,
    use_abs: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict)
    num_verifiers, _ = mat.shape

    agreements: List[float] = []
    for i in range(num_verifiers):
        others = np.delete(mat, i, axis=0)
        consensus = np.mean(others, axis=0)
        corr = safe_corr(mat[i], consensus)
        agreements.append(abs(corr) if use_abs else corr)

    w = softmax(agreements, temperature=temperature)
    weights = {names[i]: float(w[i]) for i in range(num_verifiers)}
    agree_map = {names[i]: float(agreements[i]) for i in range(num_verifiers)}
    return weights, agree_map


def weighted_candidate_scores(
    norm_scores_dict: Dict[str, Sequence[float]],
    weights: Dict[str, float],
) -> np.ndarray:
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict, verifier_names=list(weights.keys()))
    w = np.array([weights[n] for n in names], dtype=float)[:, None]
    return np.sum(w * mat, axis=0)


def votes_from_scores_threshold(
    norm_scores_dict: Dict[str, Sequence[float]],
    threshold: float = 0.5,
) -> Tuple[List[str], np.ndarray]:
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict)
    votes = (mat > float(threshold)).astype(int)
    return names, votes


def votes_from_scores_topk(
    norm_scores_dict: Dict[str, Sequence[float]],
    top_k: int = 1,
) -> Tuple[List[str], np.ndarray]:
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict)
    num_verifiers, num_candidates = mat.shape
    votes = np.zeros((num_verifiers, num_candidates), dtype=int)
    k = max(1, min(int(top_k), num_candidates))
    for i in range(num_verifiers):
        top_idx = np.argsort(-mat[i])[:k]
        votes[i, top_idx] = 1
    return names, votes


def dawid_skene_em(
    votes: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    prior: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    votes: (V, C) binary matrix in {0,1}
    Returns:
      p_true: (C,) posterior P(z=1 | votes)
      sens: (V,) sensitivity
      spec: (V,) specificity
    """
    _require_numpy()
    vote_arr = np.array(votes, dtype=int)
    num_verifiers, num_candidates = vote_arr.shape
    eps = 1e-12

    p_true = np.full(num_candidates, float(prior), dtype=float)
    sens = np.full(num_verifiers, 0.7, dtype=float)
    spec = np.full(num_verifiers, 0.7, dtype=float)

    for _ in range(int(max_iter)):
        p_old = p_true.copy()

        for c in range(num_candidates):
            ll1 = float(prior)
            ll0 = 1.0 - float(prior)
            for v in range(num_verifiers):
                y = vote_arr[v, c]
                ll1 *= sens[v] if y == 1 else (1.0 - sens[v])
                ll0 *= (1.0 - spec[v]) if y == 1 else spec[v]
            p_true[c] = ll1 / (ll1 + ll0 + eps)

        p1 = p_true
        p0 = 1.0 - p_true
        for v in range(num_verifiers):
            y = vote_arr[v].astype(float)
            sens[v] = np.sum(p1 * y) / (np.sum(p1) + eps)
            spec[v] = np.sum(p0 * (1.0 - y)) / (np.sum(p0) + eps)

        if np.max(np.abs(p_true - p_old)) < float(tol):
            break

    return p_true, sens, spec


def pairwise_verifier_correlations(
    norm_scores_dict: Dict[str, Sequence[float]]
) -> Dict[Tuple[str, str], float]:
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict)
    out: Dict[Tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[(names[i], names[j])] = safe_corr(mat[i], mat[j])
    return out


def fit_agreement_temperature(
    pool_records: Sequence[Dict[str, object]],
    global_stats: Dict[str, Dict[str, float]],
    candidate_temps: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    max_examples: int = 40,
    use_llm_judge: bool = True,
    judge_tokenizer=None,
    judge_model=None,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Select a z-score temperature by maximizing average LOO agreement.
    Uses only verifier score agreement, never correctness labels.
    """
    n = min(int(max_examples), len(pool_records))
    rows: List[Dict[str, float]] = []

    for temp in candidate_temps:
        agreement_vals: List[float] = []
        for i in range(n):
            raw = score_problem_record_raw(
                pool_records[i],
                use_llm_judge=use_llm_judge,
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
            )
            zsig, _ = normalize_scores(raw, global_stats, temperature=float(temp))
            _, agreements = loo_agreement_weights(zsig, temperature=1.0, use_abs=True)
            agreement_vals.extend(list(agreements.values()))

        avg_agree = float(sum(agreement_vals) / max(1, len(agreement_vals)))
        rows.append({"temperature": float(temp), "avg_agreement": avg_agree})

    rows.sort(key=lambda x: x["avg_agreement"], reverse=True)
    best_temp = float(rows[0]["temperature"]) if rows else 1.0
    return best_temp, rows


def _has_step_local_issue(step: str) -> bool:
    step_l = step.lower()
    if "division by zero" in step_l or "/ 0" in step_l or "÷ 0" in step_l:
        return True

    unit_words = (
        r"(apples|oranges|dollars|people|students|cars|items|cookies|"
        r"marbles|books|pencils|minutes|hours)"
    )
    if re.search(rf"\b-+\d+\s+{unit_words}\b", step_l) is not None:
        return True

    eq_match = re.search(
        r"([-+]?\d+)\s*=\s*([-+]?\d+)\s*([\+\-\*\/])\s*([-+]?\d+)",
        step.replace("×", "*"),
    )
    if not eq_match:
        return False

    lhs = int(eq_match.group(1))
    a = int(eq_match.group(2))
    op = eq_match.group(3)
    b = int(eq_match.group(4))

    if op == "+":
        return lhs != (a + b)
    if op == "-":
        return lhs != (a - b)
    if op == "*":
        return lhs != (a * b)
    if op == "/":
        return b == 0 or lhs != (a / b)
    return False


def _earliest_assignment_contradiction(steps: Sequence[str]) -> Optional[int]:
    """
    Detect repeated "X = value" statements that conflict later.
    """
    assignment_re = re.compile(r"([A-Za-z][A-Za-z0-9_ ]{0,30})=\s*([-+]?\d+)")
    first_seen: Dict[str, Tuple[int, int]] = {}
    earliest_conflict = None

    for i, step in enumerate(steps):
        for lhs, val_str in assignment_re.findall(step):
            key = re.sub(r"\s+", " ", lhs.lower().strip())
            val = int(val_str)
            if key in first_seen and first_seen[key][1] != val:
                earliest_conflict = first_seen[key][0]
                return earliest_conflict
            first_seen[key] = (i, val)
    return earliest_conflict


def pseudo_first_error_index(
    question: str,
    candidate_text: str,
    true_answer: Optional[int],
    use_llm_judge: bool = True,
    judge_threshold: float = 0.45,
    judge_tokenizer=None,
    judge_model=None,
) -> Optional[int]:
    """
    Return 0-based first wrong step index, or None if no error is detected.
    """
    pred = extract_candidate_answer(candidate_text)
    if pred is not None and true_answer is not None and int(pred) == int(true_answer):
        return None

    steps = split_into_steps(candidate_text, max_steps=16)
    if len(steps) == 0:
        return 0

    prev_steps: List[str] = []
    for i, step in enumerate(steps):
        if _has_step_local_issue(step):
            return i

        if use_llm_judge:
            score, parsed, _ = judge_step_validity(
                question=question,
                prev_steps=prev_steps,
                current_step=step,
                judge_tokenizer=judge_tokenizer,
                judge_model=judge_model,
            )
            if parsed.get("verdict", "invalid") == "invalid" or float(score) < float(judge_threshold):
                return i

        prev_steps.append(step)

    contradiction_idx = _earliest_assignment_contradiction(steps)
    if contradiction_idx is not None:
        return contradiction_idx

    return len(steps) - 1


def step_feature_map(
    question: str,
    prev_steps: Sequence[str],
    current_step: str,
    step_index: int,
    total_steps: int,
) -> Dict[str, float]:
    step_text = current_step.strip()
    prefix_text = " ".join(prev_steps)

    features = {
        "q_len": float(len(question)),
        "prefix_len": float(len(prefix_text)),
        "step_len": float(len(step_text)),
        "step_idx": float(step_index),
        "step_ratio": float(step_index + 1) / max(float(total_steps), 1.0),
        "num_count_step": float(len(extract_all_numbers(step_text))),
        "num_count_prefix": float(len(extract_all_numbers(prefix_text))),
        "has_equal": 1.0 if "=" in step_text else 0.0,
        "has_div0": 1.0 if ("/ 0" in step_text or "÷ 0" in step_text) else 0.0,
        "has_final_marker": 1.0 if "####" in step_text else 0.0,
    }
    return features


def split_by_problem_id(
    rows: Sequence[Dict[str, object]],
    train_ratio: float = 0.7,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Split rows by unique problem ID fields with configurable train/test ratio.
    """
    _require_numpy()
    problem_ids = sorted({r.get("problem_id") for r in rows})
    cut = int(train_ratio * len(problem_ids))
    train_ids = set(problem_ids[:cut])
    
    train_rows = [r for r in rows if r.get("problem_id") in train_ids]
    test_rows = [r for r in rows if r.get("problem_id") not in train_ids]
    return train_rows, test_rows


def aggregate_scores_by_problem(
    rows: Sequence[Dict[str, object]],
    pred_scores: Sequence[float],
) -> Tuple[Dict[str, List[Tuple[int, float, int]]], Dict[str, List[Tuple[int, float, int]]]]:
    """
    Aggregate per-candidate scores by problem_id.
    Returns (by_problem_with_teacher, by_problem_with_student) dicts.
    
    Each problem maps to list of (candidate_idx, score, is_correct).
    """
    by_problem = {}
    for row, score in zip(rows, pred_scores):
        pid = row.get("problem_id")
        by_problem.setdefault(pid, []).append((
            int(row.get("candidate_idx", 0)),
            float(score),
            int(row.get("is_correct", 0))
        ))
    return by_problem
