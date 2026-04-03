from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm



# =============================================================================
# Globals and shared regexes
# =============================================================================

# GSM8K answers usually end with a line like: #### 42
final_re = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")

# Generic number extractor used as a fallback
number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

# Optional global judge components for LLM-based process verification
_JUDGE_TOKENIZER = None
_JUDGE_MODEL = None


# =============================================================================
# Basic utilities
# =============================================================================

def _require_numpy() -> None:
    """Raise a helpful error if numpy is unavailable."""
    if np is None:
        raise ImportError("numpy is required for ensemble/math utilities in helpers.py")


def show_example(example: Dict[str, str]) -> None:
    """Pretty-print a GSM8K-style example."""
    print("\nQUESTION:\n", example["question"])
    print("\nSOLUTION:\n", example["answer"])
    if "true_answer" in example:
        print("\nTRUE ANSWER:", example["true_answer"])


def extract_true_answer(example_or_text: Union[Dict[str, str], str]) -> Optional[int]:
    """
    Extract the gold GSM8K answer as an integer.

    Supports either:
    - a dict like {"answer": "... #### 42"}
    - a raw answer string
    """
    if isinstance(example_or_text, dict):
        text = str(example_or_text.get("answer", ""))
    else:
        text = str(example_or_text)

    match = final_re.search(text)
    if match:
        return int(float(match.group(1)))

    # Fallback: use the last number in the text
    nums = number_re.findall(text)
    if nums:
        return int(float(nums[-1]))

    return None


def extract_pred_answer(model_text: str) -> Optional[int]:
    """
    Extract a predicted numeric answer from model output.

    Handles:
    - 'Final answer: 42'
    - 'Answer = 42'
    - GSM8K-style '#### 42'
    - fallback to the last number in the text
    """
    text = str(model_text)

    match = re.search(
        r"(?:final answer|answer)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(float(match.group(1)))

    match = final_re.search(text)
    if match:
        return int(float(match.group(1)))

    nums = number_re.findall(text)
    if nums:
        return int(float(nums[-1]))

    return None


def extract_candidate_answer(candidate_text: str) -> Optional[int]:
    """
    Backward-compatible alias.

    This keeps older notebook code working while routing to the
    main prediction extractor.
    """
    return extract_pred_answer(candidate_text)


def build_messages(
    question: str,
    system_prompt: str = "You are a helpful math tutor. Solve the problem and give the final numeric answer.",
) -> List[Dict[str, str]]:
    """Build chat-format messages for generation."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_one(
    question: str,
    model,
    tokenizer,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    top_p: float = 0.95,
    system_prompt: str = "You are a helpful math tutor. Solve the problem and give the final numeric answer.",
):
    """
    Generate one candidate solution.

    Returns:
        text, pred, input_tokens, new_tokens, total_tokens
    """
    import torch

    messages = build_messages(question, system_prompt=system_prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        outputs = model.generate(**generate_kwargs)

    gen_ids = outputs[0]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    input_tokens = inputs["input_ids"].shape[-1]
    total_tokens = gen_ids.shape[-1]
    new_tokens = max(0, total_tokens - input_tokens)

    pred = extract_pred_answer(text)
    return text, pred, int(input_tokens), int(new_tokens), int(total_tokens)


def sample_candidate_texts(
    question,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
):
    """
    Generate multiple candidate solutions for one question.

    This is the multi-sample wrapper around generate_one(...).
    """
    candidates = []

    for _ in range(n_samples):
        text, pred, in_tok, new_tok, tot_tok = generate_one(
            question,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
        )
        candidates.append(
            {
                "text": text,
                "pred": pred,
                "input_tokens": in_tok,
                "new_tokens": new_tok,
                "total_tokens": tot_tok,
            }
        )

    return candidates


# =============================================================================
# Frozen proposal pool helpers for fair Part 2 / Part 3 comparison
# =============================================================================

def build_generation_pool(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 256,
):
    """
    Generate and store one shared candidate pool per question.

    This lets multiple selection strategies operate on the exact same proposals,
    which makes comparisons fairer.
    """
    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    pool = []
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Build generation pool (n={n_samples})"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        candidates = sample_candidate_texts(
            question=q,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        total_input_tokens += sum(c["input_tokens"] for c in candidates)
        total_new_tokens += sum(c["new_tokens"] for c in candidates)
        total_tokens += sum(c["total_tokens"] for c in candidates)

        pool.append(
            {
                "idx": int(i),
                "question": q,
                "true_answer": true_a,
                "candidates": candidates,   # full text + pred + token stats
            }
        )

    pool_meta = {
        "num_questions": int(len(pool)),
        "n_samples": int(n_samples),
        "temperature": float(temperature),
        "avg_input_tokens_per_question": total_input_tokens / max(1, len(pool)),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(pool)),
        "avg_total_tokens_per_question": total_tokens / max(1, len(pool)),
    }

    return pool, pool_meta


def select_from_candidates_majority_vote(candidates):
    """
    Apply majority vote to a stored candidate list.
    """
    pred_list = [c["pred"] for c in candidates]
    final_pred = pred_list[0] if len(pred_list) == 1 else majority_vote(pred_list)

    return {
        "final_pred": final_pred,
        "selected_score": None,
        "selection_mode": "majority_vote",
        "candidate_preds": pred_list,
    }


def select_from_candidates_simple_verifier(
    candidates,
    true_answer,
    verifier_mode: str = "exact",
):
    """
    Apply a simple verifier to a stored candidate list and select the best one.
    """
    verifier_fn = get_simple_verifier_fn(verifier_mode)

    best_idx = None
    best_score = float("-inf")

    # Copy candidate dicts so we do not mutate the original stored pool
    scored_candidates = [dict(c) for c in candidates]

    for i, c in enumerate(scored_candidates):
        score = verifier_fn(c["text"], true_answer)
        scored_candidates[i]["verifier_score"] = float(score)

        if score > best_score:
            best_score = score
            best_idx = i

    return {
        "final_pred": scored_candidates[best_idx]["pred"],
        "selected_score": float(best_score),
        "selection_mode": "simple_verifier",
        "verifier_mode": verifier_mode,
        "scored_candidates": scored_candidates,
    }


def evaluate_from_generation_pool(
    generation_pool,
    pool_meta,
    selection_mode: str = "majority_vote",
    verifier_mode: str = "exact",
):
    """
    Evaluate a selection strategy on a frozen proposal pool.

    Supported selection modes:
    - majority_vote
    - simple_verifier
    """
    t0 = time.time()

    rows = []
    correct = 0

    for rec in generation_pool:
        q = rec["question"]
        true_a = rec["true_answer"]
        candidates = rec["candidates"]

        if selection_mode == "majority_vote":
            selected = select_from_candidates_majority_vote(candidates)
        elif selection_mode == "simple_verifier":
            selected = select_from_candidates_simple_verifier(
                candidates,
                true_answer=true_a,
                verifier_mode=verifier_mode,
            )
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        final_pred = selected["final_pred"]
        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)

        row = {
            "idx": int(rec["idx"]),
            "question": q,
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "n_samples": int(pool_meta["n_samples"]),
            "temperature": float(pool_meta["temperature"]),
            "selection_mode": selection_mode,
        }

        if selection_mode == "majority_vote":
            row["candidates"] = selected["candidate_preds"]
        else:
            row["selected_score"] = selected["selected_score"]
            row["verifier_mode"] = verifier_mode

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(generation_pool))

    metrics = {
        "accuracy": accuracy,
        "num_questions": int(len(generation_pool)),
        "num_correct": int(correct),
        "n_samples": int(pool_meta["n_samples"]),
        "temperature": float(pool_meta["temperature"]),
        "selection_mode": selection_mode,
        "elapsed_sec": float(elapsed),
        # Proposal-generation cost is shared across selectors
        "avg_input_tokens_per_question": float(pool_meta["avg_input_tokens_per_question"]),
        "avg_new_tokens_per_question": float(pool_meta["avg_new_tokens_per_question"]),
        "avg_total_tokens_per_question": float(pool_meta["avg_total_tokens_per_question"]),
    }

    if selection_mode == "simple_verifier":
        metrics["verifier_mode"] = verifier_mode

    return metrics, rows


def majority_vote(answers: Sequence[Optional[Union[str, int]]]) -> Optional[str]:
    """Return the most common non-null answer."""
    cleaned = [str(a) for a in answers if a is not None]
    if len(cleaned) == 0:
        return None
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


# =============================================================================
# Simple verifier functions
# =============================================================================

def verifier_exact_match(candidate_text: str, true_answer: Optional[int]) -> float:
    """Score a candidate by exact final-answer match."""
    predicted_answer = extract_candidate_answer(candidate_text)
    if predicted_answer is None or true_answer is None:
        return 0.0
    return 1.0 if predicted_answer == int(true_answer) else 0.0


def verifier_numeric_distance(candidate_text: str, true_answer: Optional[int]) -> float:
    """
    Score a candidate inversely by numeric distance from the gold answer.

    Closer answers get higher scores.
    """
    predicted_answer = extract_candidate_answer(candidate_text)
    if predicted_answer is None or true_answer is None:
        return 0.0

    distance = abs(predicted_answer - int(true_answer))
    return 1.0 / (1.0 + distance)


def verifier_format_confidence(candidate_text: str) -> float:
    """
    Heuristic confidence that text contains one clean final GSM8K answer line.
    """
    matches = final_re.findall(candidate_text)
    if len(matches) == 0:
        return 0.0
    if len(matches) > 1:
        return 0.4
    return 1.0 if "####" in candidate_text[-60:] else 0.8


def get_simple_verifier_fn(verifier_mode: str = "exact"):
    """
    Return a simple verifier function by name.

    Supported modes:
    - exact
    - numeric_distance
    - format
    """
    verifier_map = {
        "exact": verifier_exact_match,
        "numeric_distance": verifier_numeric_distance,
        # format verifier ignores true_answer, but we keep the same call shape
        "format": lambda candidate_text, true_answer: verifier_format_confidence(candidate_text),
    }

    if verifier_mode not in verifier_map:
        raise ValueError(f"Unknown verifier_mode: {verifier_mode}")

    return verifier_map[verifier_mode]


# =============================================================================
# Process-style verifier helpers
# =============================================================================

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
    """Load a tokenizer/model pair for LLM-based step judging."""
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
    """Store global judge tokenizer/model for helper reuse."""
    global _JUDGE_TOKENIZER, _JUDGE_MODEL
    _JUDGE_TOKENIZER = judge_tokenizer
    _JUDGE_MODEL = judge_model


def split_into_steps(candidate_text: str, max_steps: int = 12) -> List[str]:
    """
    Split candidate reasoning into approximate steps.

    Removes the final #### answer line first.
    """
    text = re.sub(r"####.*", "", candidate_text).strip()
    if not text:
        return []

    if "\n" in text:
        steps = [line.strip() for line in text.split("\n") if line.strip()]
    else:
        steps = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]

    return steps[:max_steps]


def build_step_judge_prompt(question: str, prev_steps: Sequence[str], current_step: str) -> str:
    """Build the prompt used for a step-validity judge."""
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
    """
    Parse JSON-like judge output into a normalized dict.
    """
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
    Judge one reasoning step and return:
    (score, parsed_json, raw_generation_text)
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

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
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
    """
    Score a candidate by averaging judge-model step-validity scores.
    """
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
    """Extract all numeric substrings from text."""
    return number_re.findall(text)


def split_reasoning_and_final(candidate_text: str) -> Tuple[str, str]:
    """Split reasoning text from the final #### answer segment."""
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
    """
    Score coherence between reasoning steps and the final predicted answer.
    """
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


# =============================================================================
# Score normalization and calibration
# =============================================================================

def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid."""
    return float(1.0 / (1.0 + math.exp(-x)))


def zscore_sigmoid(
    x: float,
    mu: float,
    sigma: float,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Normalize a scalar by z-score and squash with sigmoid.
    """
    sigma = max(float(sigma), eps)
    z = (x - mu) / sigma
    return float(sigmoid(z / max(float(temperature), eps)))


def rank_normalize(scores: Sequence[float]) -> List[float]:
    """
    Rank-normalize a list of scores to [0, 1].
    """
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
    """
    Compute raw verifier scores for a set of candidate texts.
    """
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
    """
    Build a small demo candidate set from one gold GSM8K example.
    """
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
    """
    Estimate global verifier mean/std statistics from a heldout dataset.
    """
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
    """
    Produce z-score/sigmoid and rank-normalized verifier scores.
    """
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


# =============================================================================
# Candidate-pool construction for fixed-pool experiments
# =============================================================================

def _replace_final_answer(candidate_text: str, new_value: int) -> str:
    """Replace or append a #### final answer in candidate text."""
    if final_re.search(candidate_text):
        return final_re.sub(f"#### {int(new_value)}", candidate_text)
    return candidate_text.strip() + f"\n#### {int(new_value)}"


def make_deceptive_wrong(candidate_text: str, nudge: int = 1) -> str:
    """
    Corrupt one arithmetic statement while preserving fluent structure.

    Falls back to nudging the final answer if no simple arithmetic line is found.
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
    variants.append(gold)
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
    """
    Build fixed candidate-pool records over a dataset split.
    """
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
    """Save a sequence of dict records to JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_jsonl_records(path: str) -> List[Dict[str, object]]:
    """Load JSONL records from disk."""
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
    """
    Score all candidates in one fixed-pool problem record.
    """
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
    """
    Estimate global verifier stats from fixed candidate-pool records.
    """
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


# =============================================================================
# Matrix / ensemble utilities
# =============================================================================

def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute correlation safely, returning 0 if either vector is near-constant.
    """
    _require_numpy()
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    if np.std(arr_a) < 1e-8 or np.std(arr_b) < 1e-8:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Compute softmax over a score vector."""
    _require_numpy()
    x = np.array(list(values), dtype=float) / max(float(temperature), 1e-8)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)


def dict_to_matrix(
    norm_scores_dict: Dict[str, Sequence[float]],
    verifier_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Convert a dict of verifier score lists into a matrix.

    Output shape is:
        (num_verifiers, num_candidates)
    """
    _require_numpy()
    names = list(norm_scores_dict.keys()) if verifier_names is None else list(verifier_names)
    mat = np.stack([np.array(norm_scores_dict[v], dtype=float) for v in names], axis=0)
    return names, mat


def loo_agreement_weights(
    norm_scores_dict: Dict[str, Sequence[float]],
    temperature: float = 1.0,
    use_abs: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Estimate leave-one-out agreement weights over verifier scores.
    """
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
    """Aggregate normalized verifier scores using explicit weights."""
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict, verifier_names=list(weights.keys()))
    w = np.array([weights[n] for n in names], dtype=float)[:, None]
    return np.sum(w * mat, axis=0)


def votes_from_scores_threshold(
    norm_scores_dict: Dict[str, Sequence[float]],
    threshold: float = 0.5,
) -> Tuple[List[str], np.ndarray]:
    """Convert normalized scores into binary votes using a threshold."""
    _require_numpy()
    names, mat = dict_to_matrix(norm_scores_dict)
    votes = (mat > float(threshold)).astype(int)
    return names, votes


def votes_from_scores_topk(
    norm_scores_dict: Dict[str, Sequence[float]],
    top_k: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Convert normalized scores into binary votes using per-verifier top-k selection."""
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
    Run Dawid-Skene EM on a binary vote matrix.

    votes shape:
        (num_verifiers, num_candidates)
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
    """Compute pairwise correlations across verifiers."""
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
    Select a z-score temperature by maximizing average leave-one-out agreement.

    Uses verifier agreement only, not correctness labels.
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


# =============================================================================
# First-error localization helpers
# =============================================================================

def _has_step_local_issue(step: str) -> bool:
    """Detect obvious local arithmetic or logical issues inside one step."""
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
    Detect repeated assignment statements that later contradict earlier ones.
    """
    assignment_re = re.compile(r"([A-Za-z][A-Za-z0-9_ ]{0,30})=\s*([-+]?\d+)")
    first_seen: Dict[str, Tuple[int, int]] = {}

    for i, step in enumerate(steps):
        for lhs, val_str in assignment_re.findall(step):
            key = re.sub(r"\s+", " ", lhs.lower().strip())
            val = int(val_str)
            if key in first_seen and first_seen[key][1] != val:
                return first_seen[key][0]
            first_seen[key] = (i, val)
    return None


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
    Estimate the first erroneous reasoning step in a candidate.
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
    """
    Build simple numeric features for one reasoning step.
    """
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
    Split rows by unique problem ID using a train/test ratio.
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
) -> Dict[str, List[Tuple[int, float, int]]]:
    """
    Aggregate per-candidate scores by problem ID.
    """
    by_problem = {}
    for row, score in zip(rows, pred_scores):
        pid = row.get("problem_id")
        by_problem.setdefault(pid, []).append(
            (
                int(row.get("candidate_idx", 0)),
                float(score),
                int(row.get("is_correct", 0)),
            )
        )
    return by_problem


# =============================================================================
# Unified evaluation layer for Parts 1–3
# =============================================================================

def evaluate_with_selection(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_questions: int = 100,
    max_new_tokens: int = 256,
    selection_mode: str = "majority_vote",
    verifier_mode: str = "exact",
):
    """
    General evaluation function shared by:
    - baseline
    - self-consistency
    - simple-verifier selection

    selection_mode:
    - 'majority_vote'
    - 'simple_verifier'
    """
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    rows = []
    correct = 0
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    verifier_fn = None
    if selection_mode == "simple_verifier":
        verifier_fn = get_simple_verifier_fn(verifier_mode)

    desc = "baseline" if n_samples == 1 else selection_mode

    for i in tqdm(range(len(df)), desc=f"Eval: {desc}"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        # Shared candidate-generation path for all evaluation modes
        candidates = sample_candidate_texts(
            question=q,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        total_input_tokens += sum(c["input_tokens"] for c in candidates)
        total_new_tokens += sum(c["new_tokens"] for c in candidates)
        total_tokens += sum(c["total_tokens"] for c in candidates)

        if selection_mode == "majority_vote":
            pred_list = [c["pred"] for c in candidates]
            final_pred = pred_list[0] if n_samples == 1 else majority_vote(pred_list)
            selected_score = None

        elif selection_mode == "simple_verifier":
            best_idx = None
            best_score = float("-inf")

            for j, c in enumerate(candidates):
                score = verifier_fn(c["text"], true_a)
                candidates[j]["verifier_score"] = float(score)
                if score > best_score:
                    best_score = score
                    best_idx = j

            final_pred = candidates[best_idx]["pred"]
            selected_score = float(best_score)

        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)

        row = {
            "idx": int(i),
            "question": q,
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "n_samples": int(n_samples),
            "temperature": float(temperature),
            "selection_mode": selection_mode,
        }

        if selection_mode == "majority_vote":
            row["candidates"] = [c["pred"] for c in candidates]
        else:
            row["selected_score"] = selected_score
            row["verifier_mode"] = verifier_mode

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "accuracy": accuracy,
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "n_samples": int(n_samples),
        "temperature": float(temperature),
        "selection_mode": selection_mode,
        "elapsed_sec": float(elapsed),
        "avg_input_tokens_per_question": total_input_tokens / max(1, len(df)),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(df)),
        "avg_total_tokens_per_question": total_tokens / max(1, len(df)),
    }

    if selection_mode == "simple_verifier":
        metrics["verifier_mode"] = verifier_mode

    return metrics, rows


def evaluate(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_questions: int = 100,
):
    """
    Thin wrapper for baseline and self-consistency evaluation.
    """
    return control (
        eval_df=eval_df,
        model=model,
        tokenizer=tokenizer,
        n_samples=n_samples,
        temperature=temperature,
        max_questions=max_questions,
        max_new_tokens=256,
        selection_mode="majority_vote",
    )


def best_of_n_with_simple_verifier(
    question,
    true_answer,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    verifier_mode: str = "exact",
):
    """
    Generate N candidates and select the one with the best verifier score.

    This helper is useful if you want to inspect a single question directly.
    """
    verifier_fn = get_simple_verifier_fn(verifier_mode)

    candidates = sample_candidate_texts(
        question=question,
        model=model,
        tokenizer=tokenizer,
        n_samples=n_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    best_idx = None
    best_score = float("-inf")

    for i, c in enumerate(candidates):
        score = verifier_fn(c["text"], true_answer)
        candidates[i]["verifier_score"] = float(score)

        if score > best_score:
            best_score = score
            best_idx = i

    best = candidates[best_idx]
    return best, candidates


def evaluate_best_of_n_simple_verifier(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 256,
    verifier_mode: str = "exact",
):
    """
    Thin wrapper for best-of-N evaluation with simple verifier-based selection.
    """
    return evaluate_with_selection(
        eval_df=eval_df,
        model=model,
        tokenizer=tokenizer,
        n_samples=n_samples,
        temperature=temperature,
        max_questions=max_questions,
        max_new_tokens=max_new_tokens,
        selection_mode="simple_verifier",
        verifier_mode=verifier_mode,
    )

    # =========================
# Part 4 — Difficulty Bins
# =========================

def assign_difficulty_bin(pass_rate):
    if pass_rate >= 0.8:
        return "easy"
    if pass_rate >= 0.3:
        return "medium"
    return "hard"


def estimate_pass_rate_from_pool(candidates, true_answer):
    """
    Compute empirical pass rate from a stored candidate list.
    """
    correct = 0
    total = len(candidates)

    for c in candidates:
        pred = c.get("pred")
        if pred is not None and true_answer is not None and str(pred) == str(true_answer):
            correct += 1

    return correct / max(1, total)


def build_difficulty_table_from_pool(generation_pool):
    """
    Build a difficulty table using the shared proposal pool from Parts 2 and 3.
    This avoids regenerating candidates again.
    """
    rows = []

    for rec in generation_pool:
        idx = rec["idx"]
        q = rec["question"]
        true_a = rec["true_answer"]
        candidates = rec["candidates"]

        pass_rate = estimate_pass_rate_from_pool(candidates, true_a)
        difficulty = assign_difficulty_bin(pass_rate)

        rows.append({
            "idx": int(idx),
            "question": q,
            "true_answer": true_a,
            "pass_rate": float(pass_rate),
            "difficulty": difficulty,
            "n_samples_used": int(len(candidates)),
        })

    return pd.DataFrame(rows)

    # =============================================================================
# Part 5 — Sequential revisions (self-refinement)
# =============================================================================

def build_revision_prompt(question: str, previous_attempt_text: str) -> str:
    """
    Build a revision prompt that asks the model to critique and improve
    its previous attempt.

    This is revision-specific logic only. It does not duplicate the base
    generation or extraction helpers.
    """
    return (
        "You previously attempted the following math problem.\n"
        "Your earlier reasoning may contain mistakes.\n"
        "Carefully review the previous attempt, correct any errors, "
        "and produce an improved solution.\n"
        "End your response with a final numeric answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Previous attempt:\n{previous_attempt_text}\n"
    )


def run_revision_chain(
    question,
    model,
    tokenizer,
    r_steps: int = 3,
    base_temperature: float = 0.0,
    rev_temperature: float = 0.7,
    max_new_tokens: int = 256,
):
    """
    Run a sequential revision chain:
    - attempt 0: baseline generation
    - attempts 1..R: revise previous attempt text

    Reuses:
    - generate_one(...)
    - extract_pred_answer(...) indirectly through generate_one(...)
    """
    attempts = []

    # Attempt 0: baseline
    base_text, base_pred, in_tok, new_tok, total_tok = generate_one(
        question=question,
        model=model,
        tokenizer=tokenizer,
        temperature=base_temperature,
        max_new_tokens=max_new_tokens,
    )

    attempts.append(
        {
            "attempt": 0,
            "text": base_text,
            "pred": base_pred,
            "input_tokens": in_tok,
            "new_tokens": new_tok,
            "total_tokens": total_tok,
            "attempt_type": "baseline",
        }
    )

    prev_text = base_text

    # Revision attempts
    for step in range(1, r_steps + 1):
        revision_prompt = build_revision_prompt(question, prev_text)

        revision_text, revision_pred, in_tok, new_tok, total_tok = generate_one(
            question=revision_prompt,
            model=model,
            tokenizer=tokenizer,
            temperature=rev_temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=(
                "You are a careful math tutor. Review the previous attempt, "
                "fix mistakes, and give the final numeric answer."
            ),
        )

        attempts.append(
            {
                "attempt": int(step),
                "text": revision_text,
                "pred": revision_pred,
                "input_tokens": in_tok,
                "new_tokens": new_tok,
                "total_tokens": total_tok,
                "attempt_type": "revision",
            }
        )

        prev_text = revision_text

    return attempts


def select_revision_attempt(
    attempts,
    true_answer=None,
    selection_mode: str = "last",
    verifier_mode: str = "exact",
):
    """
    Select one attempt from a revision chain.

    Supported selection modes:
    - last: choose the final revision
    - majority_vote: choose the most common predicted answer across attempts
    - simple_verifier: choose the attempt with the best verifier score
    """
    if len(attempts) == 0:
        raise ValueError("attempts must be non-empty")

    if selection_mode == "last":
        selected = dict(attempts[-1])
        selected["selection_mode"] = "last"
        return selected

    if selection_mode == "majority_vote":
        preds = [a.get("pred") for a in attempts]
        final_pred = majority_vote(preds)

        # choose the first attempt that produced that answer
        for a in attempts:
            if a.get("pred") is not None and str(a.get("pred")) == str(final_pred):
                selected = dict(a)
                selected["selection_mode"] = "majority_vote"
                selected["candidate_preds"] = preds
                return selected

        selected = dict(attempts[-1])
        selected["selection_mode"] = "majority_vote_fallback"
        selected["candidate_preds"] = preds
        return selected

    if selection_mode == "simple_verifier":
        if true_answer is None:
            raise ValueError("true_answer is required for simple_verifier selection")

        verifier_fn = get_simple_verifier_fn(verifier_mode)

        best_idx = None
        best_score = float("-inf")

        for i, attempt in enumerate(attempts):
            score = verifier_fn(attempt["text"], true_answer)
            if score > best_score:
                best_score = score
                best_idx = i

        selected = dict(attempts[best_idx])
        selected["selection_mode"] = "simple_verifier"
        selected["verifier_mode"] = verifier_mode
        selected["selected_score"] = float(best_score)
        return selected

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def evaluate_revision_chains(
    eval_df,
    model,
    tokenizer,
    r_steps: int = 3,
    base_temperature: float = 0.0,
    rev_temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 256,
    selection_mode: str = "last",
    verifier_mode: str = "exact",
):
    """
    Evaluate sequential revision chains over an eval dataframe.
    """
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    rows = []
    correct = 0
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Eval revisions: {selection_mode}"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        attempts = run_revision_chain(
            question=q,
            model=model,
            tokenizer=tokenizer,
            r_steps=r_steps,
            base_temperature=base_temperature,
            rev_temperature=rev_temperature,
            max_new_tokens=max_new_tokens,
        )

        selected = select_revision_attempt(
            attempts=attempts,
            true_answer=true_a,
            selection_mode=selection_mode,
            verifier_mode=verifier_mode,
        )

        final_pred = selected.get("pred")
        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)

        total_input_tokens += sum(a["input_tokens"] for a in attempts)
        total_new_tokens += sum(a["new_tokens"] for a in attempts)
        total_tokens += sum(a["total_tokens"] for a in attempts)

        row = {
            "idx": int(df.iloc[i]["idx"]) if "idx" in df.columns else int(i),
            "question": q,
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "r_steps": int(r_steps),
            "base_temperature": float(base_temperature),
            "rev_temperature": float(rev_temperature),
            "selection_mode": selection_mode,
            "attempts": attempts,
        }

        if "difficulty" in df.columns:
            row["difficulty"] = df.iloc[i]["difficulty"]
        if "pass_rate" in df.columns:
            row["pass_rate"] = df.iloc[i]["pass_rate"]

        if selection_mode == "simple_verifier":
            row["selected_score"] = selected.get("selected_score")
            row["verifier_mode"] = verifier_mode

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "method": "sequential_revisions",
        "accuracy": float(accuracy),
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "r_steps": int(r_steps),
        "base_temperature": float(base_temperature),
        "rev_temperature": float(rev_temperature),
        "selection_mode": selection_mode,
        "elapsed_sec": float(elapsed),
        "avg_input_tokens_per_question": total_input_tokens / max(1, len(df)),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(df)),
        "avg_total_tokens_per_question": total_tokens / max(1, len(df)),
    }

    if selection_mode == "simple_verifier":
        metrics["verifier_mode"] = verifier_mode

    return metrics, rows


# =============================================================================
# Part 6 — Lightweight beam / search
# =============================================================================

def build_step_prompt(question: str, partial_reasoning: str) -> str:
    """
    Build a prompt for one-step beam expansion.

    This is beam-specific prompt logic only.
    """
    partial_reasoning = (partial_reasoning or "").strip()

    return (
        "You are solving a math word problem step by step.\n"
        "Write the next useful reasoning step only.\n"
        "If you are ready to finish, end with the final numeric answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Current partial reasoning:\n{partial_reasoning if partial_reasoning else '(start)'}\n"
    )


def generate_continuations_with_scores(
    question,
    model,
    tokenizer,
    num_return_sequences: int = 2,
    temperature: float = 0.7,
    max_new_tokens: int = 80,
    system_prompt: str = (
        "You are a careful math tutor. Continue the reasoning by one useful step "
        "or provide the final numeric answer if ready."
    ),
):
    """
    Generate multiple continuations for a single prompt and compute
    average token log-probability for each continuation.

    This reuses the same chat-template approach as the rest of the notebook,
    but adds output_scores=True so Part 6 can rank beam expansions.
    """
    import torch

    messages = build_messages(question, system_prompt=system_prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "num_return_sequences": int(num_return_sequences),
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        outputs = model.generate(**generate_kwargs)

    sequences = outputs.sequences
    scores_per_step = outputs.scores
    prompt_len = inputs["input_ids"].shape[-1]

    results = []

    for seq_idx in range(sequences.shape[0]):
        full_ids = sequences[seq_idx]
        gen_ids = full_ids[prompt_len:]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        gen_tokens = int(gen_ids.shape[-1])

        avg_logprob = 0.0
        if gen_tokens > 0:
            step_logps = []
            for t in range(gen_tokens):
                step_scores = scores_per_step[t][seq_idx]
                log_probs = torch.log_softmax(step_scores, dim=-1)
                tok_id = int(gen_ids[t].item())
                step_logps.append(float(log_probs[tok_id].item()))
            avg_logprob = float(sum(step_logps) / len(step_logps))

        pred = extract_pred_answer(gen_text)
        results.append(
            {
                "text": full_text,
                "generated_text": gen_text,
                "pred": pred,
                "avg_logprob": avg_logprob,
                "input_tokens": int(prompt_len),
                "new_tokens": int(gen_tokens),
                "total_tokens": int(full_ids.shape[-1]),
            }
        )

    return results


def beam_search_reasoning(
    question,
    model,
    tokenizer,
    true_answer=None,
    beam_size: int = 3,
    expansions_per_beam: int = 2,
    max_steps: int = 4,
    temperature: float = 0.7,
    step_max_new_tokens: int = 80,
    length_penalty: float = 0.02,
    verifier_mode: str = "exact",
):
    """
    Lightweight beam-style search over partial reasoning.

    Each beam item stores:
    - partial reasoning text
    - accumulated score
    - token count
    - whether it appears finished
    - extracted predicted answer if available
    """
    beam = [
        {
            "partial_text": "",
            "score": 0.0,
            "steps": 0,
            "total_new_tokens": 0,
            "finished": False,
            "pred": None,
        }
    ]

    finished = []

    for step_idx in range(max_steps):
        expanded = []

        for state in beam:
            if state["finished"]:
                expanded.append(dict(state))
                continue

            step_prompt = build_step_prompt(question, state["partial_text"])

            continuations = generate_continuations_with_scores(
                question=step_prompt,
                model=model,
                tokenizer=tokenizer,
                num_return_sequences=expansions_per_beam,
                temperature=temperature,
                max_new_tokens=step_max_new_tokens,
            )

            for cont in continuations:
                prev_partial = state["partial_text"].strip()
                continuation = cont["generated_text"].strip()

                if prev_partial and continuation:
                    new_partial = prev_partial + "\n" + continuation
                else:
                    new_partial = prev_partial or continuation

                total_new_tokens = state["total_new_tokens"] + cont["new_tokens"]

                new_score = (
                    float(state["score"])
                    + float(cont["avg_logprob"])
                    - float(length_penalty) * float(total_new_tokens)
                )

                pred = extract_pred_answer(new_partial)
                finished_flag = pred is not None

                expanded.append(
                    {
                        "partial_text": new_partial,
                        "score": float(new_score),
                        "steps": int(step_idx + 1),
                        "total_new_tokens": int(total_new_tokens),
                        "finished": bool(finished_flag),
                        "pred": pred,
                    }
                )

        beam = sorted(expanded, key=lambda x: x["score"], reverse=True)[:beam_size]

        for state in beam:
            if state["finished"]:
                finished.append(dict(state))

        if len(finished) >= beam_size:
            break

    candidate_pool = finished if len(finished) > 0 else beam

    if true_answer is not None:
        verifier_fn = get_simple_verifier_fn(verifier_mode)

        best_idx = None
        best_verifier_score = float("-inf")

        for i, state in enumerate(candidate_pool):
            score = verifier_fn(state["partial_text"], true_answer)
            if score > best_verifier_score:
                best_verifier_score = score
                best_idx = i

        best = dict(candidate_pool[best_idx])
        best["selected_score"] = float(best_verifier_score)
        best["verifier_mode"] = verifier_mode
        best["used_pool"] = "finished" if len(finished) > 0 else "beam"
        return best

    return dict(candidate_pool[0])


def evaluate_beam_search(
    eval_df,
    model,
    tokenizer,
    beam_size: int = 3,
    expansions_per_beam: int = 2,
    max_steps: int = 4,
    temperature: float = 0.7,
    step_max_new_tokens: int = 80,
    length_penalty: float = 0.02,
    verifier_mode: str = "exact",
    max_questions: int = 30,
):
    """
    Evaluate beam-search style reasoning over an evaluation dataframe.
    """
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    rows = []
    correct = 0
    total_new_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Eval beam search (B={beam_size})"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        selected = beam_search_reasoning(
            question=q,
            model=model,
            tokenizer=tokenizer,
            true_answer=true_a,
            beam_size=beam_size,
            expansions_per_beam=expansions_per_beam,
            max_steps=max_steps,
            temperature=temperature,
            step_max_new_tokens=step_max_new_tokens,
            length_penalty=length_penalty,
            verifier_mode=verifier_mode,
        )

        final_pred = selected.get("pred")
        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)
        total_new_tokens += int(selected.get("total_new_tokens", 0))

        row = {
            "idx": int(df.iloc[i]["idx"]) if "idx" in df.columns else int(i),
            "question": q,
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "beam_size": int(beam_size),
            "expansions_per_beam": int(expansions_per_beam),
            "max_steps": int(max_steps),
            "score": float(selected.get("score", 0.0)),
            "selected_score": float(selected.get("selected_score", 0.0)),
            "verifier_mode": verifier_mode,
            "used_pool": selected.get("used_pool"),
            "total_new_tokens": int(selected.get("total_new_tokens", 0)),
            "text": selected.get("partial_text", ""),
        }

        if "difficulty" in df.columns:
            row["difficulty"] = df.iloc[i]["difficulty"]
        if "pass_rate" in df.columns:
            row["pass_rate"] = df.iloc[i]["pass_rate"]

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "method": "beam_search",
        "accuracy": float(accuracy),
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "beam_size": int(beam_size),
        "expansions_per_beam": int(expansions_per_beam),
        "max_steps": int(max_steps),
        "temperature": float(temperature),
        "step_max_new_tokens": int(step_max_new_tokens),
        "length_penalty": float(length_penalty),
        "verifier_mode": verifier_mode,
        "elapsed_sec": float(elapsed),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(df)),
    }

    return metrics, rows

# =============================================================================
# Part 7 — Difficulty-aware routing policy
# =============================================================================

def run_routed_strategy(
    question,
    true_answer,
    model,
    tokenizer,
    strategy_name: str,
    cfg: dict,
):
    """
    Run one already-defined TTC strategy under a common interface.

    Supported strategies:
    - S1: baseline single-pass
    - S2: best-of-N majority vote
    - S3: sequential revisions
    - S4: beam search

    Returns a normalized result dict with:
    - pred
    - text
    - total_new_tokens
    - strategy
    """
    strategy_name = str(strategy_name)

    if strategy_name == "S1":
        text, pred, _, new_tok, _ = generate_one(
            question=question,
            model=model,
            tokenizer=tokenizer,
            temperature=cfg.get("temperature", 0.0),
            max_new_tokens=cfg.get("max_new_tokens", 256),
        )
        return {
            "pred": pred,
            "text": text,
            "total_new_tokens": int(new_tok),
            "strategy": "S1",
        }

    if strategy_name == "S2":
        n_samples = int(cfg.get("n_samples", 5))
        temperature = float(cfg.get("temperature", 0.7))
        max_new_tokens = int(cfg.get("max_new_tokens", 256))

        candidates = sample_candidate_texts(
            question=question,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        pred_list = [c["pred"] for c in candidates]
        final_pred = pred_list[0] if len(pred_list) == 1 else majority_vote(pred_list)

        chosen_text = ""
        for c in candidates:
            if c["pred"] is not None and str(c["pred"]) == str(final_pred):
                chosen_text = c["text"]
                break

        return {
            "pred": final_pred,
            "text": chosen_text,
            "total_new_tokens": int(sum(c["new_tokens"] for c in candidates)),
            "strategy": "S2",
        }

    if strategy_name == "S3":
        attempts = run_revision_chain(
            question=question,
            model=model,
            tokenizer=tokenizer,
            r_steps=int(cfg.get("r_steps", 2)),
            base_temperature=float(cfg.get("base_temperature", 0.0)),
            rev_temperature=float(cfg.get("rev_temperature", 0.7)),
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        )

        selected = select_revision_attempt(
            attempts=attempts,
            true_answer=true_answer,
            selection_mode=cfg.get("selection_mode", "simple_verifier"),
            verifier_mode=cfg.get("verifier_mode", "exact"),
        )

        return {
            "pred": selected.get("pred"),
            "text": selected.get("text", ""),
            "total_new_tokens": int(sum(a["new_tokens"] for a in attempts)),
            "strategy": "S3",
        }

    if strategy_name == "S4":
        selected = beam_search_reasoning(
            question=question,
            model=model,
            tokenizer=tokenizer,
            true_answer=true_answer,
            beam_size=int(cfg.get("beam_size", 3)),
            expansions_per_beam=int(cfg.get("expansions_per_beam", 2)),
            max_steps=int(cfg.get("max_steps", 4)),
            temperature=float(cfg.get("temperature", 0.7)),
            step_max_new_tokens=int(cfg.get("step_max_new_tokens", 80)),
            length_penalty=float(cfg.get("length_penalty", 0.02)),
            verifier_mode=cfg.get("verifier_mode", "exact"),
        )

        return {
            "pred": selected.get("pred"),
            "text": selected.get("partial_text", ""),
            "total_new_tokens": int(selected.get("total_new_tokens", 0)),
            "strategy": "S4",
        }

    raise ValueError(f"Unknown strategy_name: {strategy_name}")


def evaluate_routing_policy(
    eval_df,
    model,
    tokenizer,
    policy_map: dict,
    cfg_map: dict,
    max_questions: int = 30,
    policy_name: str = "policy",
):
    """
    Evaluate a difficulty-aware routing policy.

    policy_map example:
        {"easy": "S1", "medium": "S2", "hard": "S3"}

    cfg_map example:
        {
            "S1": {...},
            "S2": {...},
            "S3": {...},
            "S4": {...},
        }
    """
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    rows = []
    correct = 0
    total_new_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Eval routing: {policy_name}"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])
        difficulty = df.iloc[i]["difficulty"] if "difficulty" in df.columns else "medium"

        strategy_name = policy_map.get(difficulty, "S2")
        cfg = cfg_map.get(strategy_name, {})

        out = run_routed_strategy(
            question=q,
            true_answer=true_a,
            model=model,
            tokenizer=tokenizer,
            strategy_name=strategy_name,
            cfg=cfg,
        )

        final_pred = out.get("pred")
        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)
        total_new_tokens += int(out.get("total_new_tokens", 0))

        row = {
            "idx": int(df.iloc[i]["idx"]) if "idx" in df.columns else int(i),
            "question": q,
            "true_answer": true_a,
            "difficulty": difficulty,
            "strategy": strategy_name,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "total_new_tokens": int(out.get("total_new_tokens", 0)),
            "text": out.get("text", ""),
        }

        if "pass_rate" in df.columns:
            row["pass_rate"] = df.iloc[i]["pass_rate"]

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "method": "difficulty_aware_routing",
        "policy_name": policy_name,
        "accuracy": float(accuracy),
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(df)),
        "elapsed_sec": float(elapsed),
        "policy_map": dict(policy_map),
    }

    return metrics, rows


def summarize_policy_results(policy_results: dict):
    """
    Convert a dict like:
        {"A": metricsA, "B": metricsB, ...}
    into a compact dataframe for comparison.
    """
    rows = []
    for policy_name, metrics in policy_results.items():
        rows.append(
            {
                "policy": policy_name,
                "accuracy": metrics.get("accuracy"),
                "num_correct": metrics.get("num_correct"),
                "num_questions": metrics.get("num_questions"),
                "avg_new_tokens_per_question": metrics.get("avg_new_tokens_per_question"),
                "elapsed_sec": metrics.get("elapsed_sec"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["accuracy", "avg_new_tokens_per_question"],
        ascending=[False, True],
    )

