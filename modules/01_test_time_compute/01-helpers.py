from __future__ import annotations

"""Utilities for Module 1 test-time compute experiments.

"""

import re
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm


# GSM8K answers usually end with a line like: #### 42
final_re = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")

# Generic number extractor used as a fallback
number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")


def set_random_seed(seed: int) -> None:
    """Best-effort seeding for reproducible notebook runs."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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
    """Backward-compatible alias for prediction extraction."""
    return extract_pred_answer(candidate_text)


def render_prompt_template(template: str, **kwargs: str) -> str:
    """Render a prompt template supplied by the notebook."""
    if template is None:
        raise ValueError("Prompt templates must be defined in the notebook and passed explicitly.")

    try:
        return str(template).format(**kwargs)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Prompt template is missing required placeholder: {missing_key}") from exc


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    """Build chat-format messages for generation."""
    if system_prompt is None:
        raise ValueError("system_prompt must be defined in the notebook and passed explicitly.")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_one(
    question: str,
    model,
    tokenizer,
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    top_p: float = 0.95,
    system_prompt: str = None,
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

    input_tokens = inputs["input_ids"].shape[-1]
    total_tokens = gen_ids.shape[-1]
    new_tokens = max(0, total_tokens - input_tokens)
    generated_ids = gen_ids[input_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    pred = extract_pred_answer(text)
    return text, pred, int(input_tokens), int(new_tokens), int(total_tokens)


def sample_candidate_texts(
    question,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 1280,
    system_prompt: str = None,
):
    """Generate multiple candidate solutions for one question."""
    candidates = []

    for _ in range(n_samples):
        text, pred, in_tok, new_tok, tot_tok = generate_one(
            question,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            system_prompt=system_prompt,
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


def compute_token_summary(
    total_input_tokens: float,
    total_new_tokens: float,
    total_tokens: float,
    num_items: int,
) -> Dict[str, float]:
    """Build a consistent token-cost summary across evaluation methods."""
    denom = max(1, int(num_items))
    return {
        "avg_tokens": float(total_tokens / denom),
        "avg_input_tokens_per_question": float(total_input_tokens / denom),
        "avg_new_tokens_per_question": float(total_new_tokens / denom),
        "avg_total_tokens_per_question": float(total_tokens / denom),
    }


def build_generation_pool(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 1280,
    system_prompt: str = None,
):
    """
    Generate and store one shared candidate pool per question.

    This lets multiple selection strategies operate on the exact same answers,
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
            system_prompt=system_prompt,
        )

        total_input_tokens += sum(c["input_tokens"] for c in candidates)
        total_new_tokens += sum(c["new_tokens"] for c in candidates)
        total_tokens += sum(c["total_tokens"] for c in candidates)

        pool.append(
            {
                "idx": int(i),
                "question": q,
                "true_answer": true_a,
                "candidates": candidates,
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
    """Apply majority vote to a stored candidate list."""
    pred_list = [c["pred"] for c in candidates]
    final_pred = pred_list[0] if len(pred_list) == 1 else majority_vote(pred_list)

    return {
        "final_pred": final_pred,
        "selected_score": None,
        "selection_mode": "majority_vote",
        "candidate_preds": pred_list,
    }


def evaluate_from_generation_pool(
    generation_pool,
    pool_meta,
    selection_mode: str = "majority_vote",
):
    """
    Evaluate a selection strategy on a frozen answer pool.

    Supported selection modes:
    - majority_vote
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

        row["candidates"] = selected["candidate_preds"]

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(generation_pool))

    metrics = {
        "method": "self_consistency",
        "accuracy": accuracy,
        "num_questions": int(len(generation_pool)),
        "num_correct": int(correct),
        "n_samples": int(pool_meta["n_samples"]),
        "temperature": float(pool_meta["temperature"]),
        "selection_mode": selection_mode,
        "elapsed_sec": float(elapsed),
    }
    metrics.update(
        compute_token_summary(
            total_input_tokens=pool_meta["avg_input_tokens_per_question"] * len(generation_pool),
            total_new_tokens=pool_meta["avg_new_tokens_per_question"] * len(generation_pool),
            total_tokens=pool_meta["avg_total_tokens_per_question"] * len(generation_pool),
            num_items=len(generation_pool),
        )
    )

    return metrics, rows


def majority_vote(answers: Sequence[Optional[Union[str, int]]]) -> Optional[str]:
    """Return the most common non-null answer."""
    cleaned = [str(a) for a in answers if a is not None]
    if len(cleaned) == 0:
        return None
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def evaluate_with_selection(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_questions: int = 100,
    max_new_tokens: int = 1280,
    system_prompt: str = None,
    selection_mode: str = "majority_vote",
):
    """
    General evaluation function shared by:
    - baseline
    - self-consistency
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

    desc = "baseline" if n_samples == 1 else selection_mode

    for i in tqdm(range(len(df)), desc=f"Eval: {desc}"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        candidates = sample_candidate_texts(
            question=q,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )

        total_input_tokens += sum(c["input_tokens"] for c in candidates)
        total_new_tokens += sum(c["new_tokens"] for c in candidates)
        total_tokens += sum(c["total_tokens"] for c in candidates)

        if selection_mode == "majority_vote":
            pred_list = [c["pred"] for c in candidates]
            final_pred = pred_list[0] if n_samples == 1 else majority_vote(pred_list)
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

        row["candidates"] = [c["pred"] for c in candidates]

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "method": "baseline" if n_samples == 1 else "self_consistency",
        "accuracy": accuracy,
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "n_samples": int(n_samples),
        "temperature": float(temperature),
        "selection_mode": selection_mode,
        "elapsed_sec": float(elapsed),
    }
    metrics.update(
        compute_token_summary(
            total_input_tokens=total_input_tokens,
            total_new_tokens=total_new_tokens,
            total_tokens=total_tokens,
            num_items=len(df),
        )
    )

    return metrics, rows


def evaluate(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_questions: int = 100,
    max_new_tokens: int = 1280,
    system_prompt: str = None,
):
    """Thin wrapper for baseline and self-consistency evaluation."""
    return evaluate_with_selection(
        eval_df=eval_df,
        model=model,
        tokenizer=tokenizer,
        n_samples=n_samples,
        temperature=temperature,
        max_questions=max_questions,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        selection_mode="majority_vote",
    )

def assign_difficulty_bin(pass_rate):
    if pass_rate >= 0.8:
        return "easy"
    if pass_rate >= 0.3:
        return "medium"
    return "hard"


def estimate_pass_rate_from_pool(candidates, true_answer):
    """Compute empirical pass rate from a stored candidate list."""
    correct = 0
    total = len(candidates)

    for candidate in candidates:
        pred = candidate.get("pred")
        if pred is not None and true_answer is not None and str(pred) == str(true_answer):
            correct += 1

    return correct / max(1, total)


def build_difficulty_table_from_pool(generation_pool):
    """
    Build a difficulty table using the shared answer pool from the sampling phase.

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

        rows.append(
            {
                "idx": int(idx),
                "question": q,
                "true_answer": true_a,
                "pass_rate": float(pass_rate),
                "difficulty": difficulty,
                "n_samples_used": int(len(candidates)),
            }
        )

    return pd.DataFrame(rows)


def prepare_eval_with_difficulty(eval_df, difficulty_df):
    """
    Attach difficulty labels back onto the eval subset used to build the pool.

    The returned dataframe is intended for descriptive analysis of TTC behavior
    by difficulty, not for training or validating a downstream decision rule.
    """
    eval_with_diff = eval_df.iloc[: len(difficulty_df)].copy().reset_index()
    eval_with_diff = eval_with_diff.rename(columns={"index": "idx"})
    return eval_with_diff.merge(
        difficulty_df[["idx", "difficulty", "pass_rate"]],
        on="idx",
        how="left",
    )


def split_eval_dataframe(
    eval_df: pd.DataFrame,
    primary_size: int,
    secondary_size: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an eval dataframe into two disjoint sequential subsets."""
    primary_df = eval_df.iloc[:primary_size].copy().reset_index(drop=True)
    remainder = eval_df.iloc[primary_size:].copy().reset_index(drop=True)
    if secondary_size is not None:
        remainder = remainder.iloc[:secondary_size].copy().reset_index(drop=True)
    return primary_df, remainder


def build_revision_prompt(
    question: str,
    previous_attempt_text: str,
    revision_prompt_template: str,
) -> str:
    """Build a revision prompt from the notebook template."""
    return render_prompt_template(
        revision_prompt_template,
        question=question,
        previous_attempt_text=previous_attempt_text,
    )


def run_revision_chain(
    question,
    model,
    tokenizer,
    r_steps: int = 3,
    base_temperature: float = 0.0,
    rev_temperature: float = 0.7,
    max_new_tokens: int = 1280,
    base_system_prompt: str = None,
    revision_prompt_template: str = None,
    revision_system_prompt: str = None,
):
    """
    Run a sequential revision chain:
    - attempt 0: baseline generation
    - attempts 1..R: revise previous attempt text
    """
    attempts = []

    base_text, base_pred, in_tok, new_tok, total_tok = generate_one(
        question=question,
        model=model,
        tokenizer=tokenizer,
        temperature=base_temperature,
        max_new_tokens=max_new_tokens,
        system_prompt=base_system_prompt,
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

    for step in range(1, r_steps + 1):
        revision_prompt = build_revision_prompt(
            question,
            prev_text,
            revision_prompt_template=revision_prompt_template,
        )

        revision_text, revision_pred, in_tok, new_tok, total_tok = generate_one(
            question=revision_prompt,
            model=model,
            tokenizer=tokenizer,
            temperature=rev_temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=revision_system_prompt,
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
    selection_mode: str = "last",
):
    """
    Select one attempt from a revision chain.

    Supported selection modes:
    - last
    - majority_vote
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

        for attempt in attempts:
            if attempt.get("pred") is not None and str(attempt.get("pred")) == str(final_pred):
                selected = dict(attempt)
                selected["selection_mode"] = "majority_vote"
                selected["candidate_preds"] = preds
                return selected

        selected = dict(attempts[-1])
        selected["selection_mode"] = "majority_vote_fallback"
        selected["candidate_preds"] = preds
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
    max_new_tokens: int = 1280,
    base_system_prompt: str = None,
    revision_prompt_template: str = None,
    revision_system_prompt: str = None,
    selection_mode: str = "last",
):
    """Evaluate sequential revision chains over an eval dataframe."""
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
            base_system_prompt=base_system_prompt,
            revision_prompt_template=revision_prompt_template,
            revision_system_prompt=revision_system_prompt,
        )

        selected = select_revision_attempt(
            attempts=attempts,
            selection_mode=selection_mode,
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
    }
    metrics.update(
        compute_token_summary(
            total_input_tokens=total_input_tokens,
            total_new_tokens=total_new_tokens,
            total_tokens=total_tokens,
            num_items=len(df),
        )
    )

    return metrics, rows


def build_step_prompt(question: str, partial_reasoning: str, step_prompt_template: str) -> str:
    """Build a prompt for one-step beam expansion."""
    partial_reasoning = (partial_reasoning or "").strip()

    return render_prompt_template(
        step_prompt_template,
        question=question,
        partial_reasoning=partial_reasoning if partial_reasoning else "(start)",
    )


def generate_continuations_with_scores(
    question,
    model,
    tokenizer,
    num_return_sequences: int = 2,
    temperature: float = 0.7,
    max_new_tokens: int = 80,
    system_prompt: str = None,
):
    """
    Generate multiple continuations for a single prompt and compute
    average token log-probability for each continuation.
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
    beam_size: int = 3,
    expansions_per_beam: int = 2,
    max_steps: int = 4,
    temperature: float = 0.7,
    step_max_new_tokens: int = 80,
    length_penalty: float = 0.02,
    step_prompt_template: str = None,
    step_system_prompt: str = None,
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
            "total_tokens_spent": 0,
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

            step_prompt = build_step_prompt(
                question,
                state["partial_text"],
                step_prompt_template=step_prompt_template,
            )

            continuations = generate_continuations_with_scores(
                question=step_prompt,
                model=model,
                tokenizer=tokenizer,
                num_return_sequences=expansions_per_beam,
                temperature=temperature,
                max_new_tokens=step_max_new_tokens,
                system_prompt=step_system_prompt,
            )

            for cont in continuations:
                prev_partial = state["partial_text"].strip()
                continuation = cont["generated_text"].strip()

                if prev_partial and continuation:
                    new_partial = prev_partial + "\n" + continuation
                else:
                    new_partial = prev_partial or continuation

                total_new_tokens = state["total_new_tokens"] + cont["new_tokens"]
                total_tokens_spent = state["total_tokens_spent"] + cont["total_tokens"]

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
                        "total_tokens_spent": int(total_tokens_spent),
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
    candidate_pool = sorted(candidate_pool, key=lambda x: x["score"], reverse=True)
    best = dict(candidate_pool[0])
    best["used_pool"] = "finished" if len(finished) > 0 else "beam"
    return best


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
    max_questions: int = 30,
    step_prompt_template: str = None,
    step_system_prompt: str = None,
):
    """Evaluate beam-search style reasoning over an evaluation dataframe."""
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    rows = []
    correct = 0
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Eval beam search (B={beam_size})"):
        q = df.iloc[i]["question"]
        true_a = extract_true_answer(df.iloc[i]["answer"])

        selected = beam_search_reasoning(
            question=q,
            model=model,
            tokenizer=tokenizer,
            beam_size=beam_size,
            expansions_per_beam=expansions_per_beam,
            max_steps=max_steps,
            temperature=temperature,
            step_max_new_tokens=step_max_new_tokens,
            length_penalty=length_penalty,
            step_prompt_template=step_prompt_template,
            step_system_prompt=step_system_prompt,
        )

        final_pred = selected.get("pred")
        is_correct = (
            final_pred is not None
            and true_a is not None
            and str(final_pred) == str(true_a)
        )
        correct += int(is_correct)
        total_input_tokens += int(selected.get("total_tokens_spent", 0) - selected.get("total_new_tokens", 0))
        total_new_tokens += int(selected.get("total_new_tokens", 0))
        total_tokens += int(selected.get("total_tokens_spent", 0))

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
            "used_pool": selected.get("used_pool"),
            "total_new_tokens": int(selected.get("total_new_tokens", 0)),
            "total_tokens_spent": int(selected.get("total_tokens_spent", 0)),
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
        "elapsed_sec": float(elapsed),
    }
    metrics.update(
        compute_token_summary(
            total_input_tokens=total_input_tokens,
            total_new_tokens=total_new_tokens,
            total_tokens=total_tokens,
            num_items=len(df),
        )
    )

    return metrics, rows
