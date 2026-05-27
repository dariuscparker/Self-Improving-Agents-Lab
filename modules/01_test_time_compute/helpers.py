from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Union
import pandas as pd
from tqdm import tqdm



# =============================================================================
# Globals and shared regexes
# =============================================================================

# GSM8K answers usually end with a line like: #### 42
final_re = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")

# Generic number extractor used as a fallback
number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

# =============================================================================
# Basic utilities
# =============================================================================

NOTEBOOK_EXPORTS = [
    "final_re",
    "extract_true_answer",
    "show_example",
    "build_token_preview",
    "show_token_metrics_preview",
    "show_token_human_readable_preview",
    "evaluate",
    "build_generation_pool",
    "evaluate_from_generation_pool",
    "build_difficulty_table_from_pool",
    "attach_difficulty_columns",
    "summarize_difficulty_bin_performance",
    "summarize_result_rows_by_difficulty",
    "build_revision_before_after_summary",
    "build_revision_chain_pool",
    "evaluate_from_revision_chain_pool",
    "build_experiment_payload",
    "build_metrics_table",
    "metrics_summary_view",
    "plot_accuracy_vs_compute",
    "write_results_summary",
    "write_json_artifact",
]


def load_notebook_helpers(namespace: dict) -> None:
    """Expose the small public helper surface used by the notebook."""
    namespace.update({name: globals()[name] for name in NOTEBOOK_EXPORTS})


def show_example(example: Dict[str, str]) -> None:
    """Pretty-print a GSM8K-style example."""
    print("\nQUESTION:\n", example["question"])
    print("\nSOLUTION:\n", example["answer"])
    if "true_answer" in example:
        print("\nTRUE ANSWER:", example["true_answer"])


def _final_output_after_question(text: str, question: str) -> str:
    """Keep the generated answer text without the repeated prompt when possible."""
    return text.split(question, 1)[-1].strip() if question in text else text.strip()


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


def answers_match(predicted, true_answer) -> bool:
    """Compare extracted numeric answers while handling missing values."""
    return (
        predicted is not None
        and true_answer is not None
        and str(predicted) == str(true_answer)
    )


def sum_token_counts(records):
    """Sum token accounting fields across generated candidates or attempts."""
    return {
        "input_tokens": int(sum(r["input_tokens"] for r in records)),
        "new_tokens": int(sum(r["new_tokens"] for r in records)),
        "total_tokens": int(sum(r["total_tokens"] for r in records)),
    }


def build_messages(
    question: str,
    system_prompt: str | None = None,
) -> List[Dict[str, str]]:
    """Build chat-format messages for generation."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def generate_one(
    question: str,
    model,
    tokenizer,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    top_p: float = 0.95,
    system_prompt: str | None = None,
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

    input_tokens = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0]
    completion_ids = gen_ids[input_tokens:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    total_tokens = gen_ids.shape[-1]
    new_tokens = max(0, total_tokens - input_tokens)

    pred = extract_pred_answer(text)
    return text, pred, int(input_tokens), int(new_tokens), int(total_tokens)


def build_token_preview(
    eval_df,
    model,
    tokenizer,
    n: int = 5,
    random_state: int = 0,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    system_prompt: str | None = None,
):
    """Generate a small sample preview and summarize output-token counts."""
    preview_df = eval_df.sample(n=min(n, len(eval_df)), random_state=random_state).copy()
    preview_df["idx"] = preview_df.index
    preview_df = preview_df.reset_index(drop=True)

    rows = []
    for _, row in tqdm(preview_df.iterrows(), total=len(preview_df), desc=f"{n}-sample preview"):
        question = row["question"]
        true_answer = extract_true_answer(row["answer"])
        text, pred, input_tokens, output_tokens, total_tokens = generate_one(
            question=question,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )

        rows.append(
            {
                "idx": int(row["idx"]),
                "question": question,
                "final_output": _final_output_after_question(text, question),
                "true_answer": true_answer,
                "predicted_answer": pred,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "correct": pred == true_answer,
            }
        )

    preview = pd.DataFrame(rows)
    token_stats = pd.DataFrame(
        [
            {"metric": "low_output_tokens", "tokens": int(preview["output_tokens"].min())},
            {"metric": "avg_output_tokens", "tokens": float(preview["output_tokens"].mean())},
            {"metric": "high_output_tokens", "tokens": int(preview["output_tokens"].max())},
        ]
    )

    return preview, token_stats


def show_token_metrics_preview(preview_df, token_stats) -> None:
    """Display token preview summary tables."""
    from IPython.display import display

    print("Five-sample output-token EDA:")
    display(token_stats)

    summary_cols = [
        "idx",
        "output_tokens",
        "input_tokens",
        "total_tokens",
        "correct",
        "true_answer",
        "predicted_answer",
    ]
    display(preview_df[summary_cols])


def show_token_human_readable_preview(preview_df, example_index: int = 0) -> None:
    """Print one human-readable token-preview example."""
    example = preview_df.iloc[example_index]
    print("\nHuman-readable example")
    print("=" * 80)
    print(f"Preview row: {example_index}")
    print(f"Question idx: {example['idx']}")
    print(f"Output tokens: {example['output_tokens']}")
    print(f"Correct: {example['correct']}")
    print(f"True answer: {example['true_answer']}")
    print(f"Predicted answer: {example['predicted_answer']}")
    print("\nQuestion:\n")
    print(example["question"])
    print("\nFinal output:\n")
    print(example["final_output"])


def sample_candidate_texts(
    question,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    system_prompt: str | None = None,
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


# =============================================================================
# Frozen proposal pool helpers
# =============================================================================

def build_generation_pool(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 256,
    system_prompt: str | None = None,
):
    """
    Generate and store one frozen candidate pool per question.

    This lets later analysis reuse the exact same generated proposals instead
    of mixing results with newly sampled generations.
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
        token_counts = sum_token_counts(candidates)

        total_input_tokens += token_counts["input_tokens"]
        total_new_tokens += token_counts["new_tokens"]
        total_tokens += token_counts["total_tokens"]

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
        "system_prompt": system_prompt,
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


def select_from_candidates(
    candidates,
    selection_mode: str = "majority_vote",
):
    """Select one answer from generated candidates using a named strategy."""
    if selection_mode == "majority_vote":
        return select_from_candidates_majority_vote(candidates)

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def evaluate_from_generation_pool(
    generation_pool,
    pool_meta,
    selection_mode: str = "majority_vote",
):
    """
    Evaluate a selection strategy on a frozen proposal pool.

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

        selected = select_from_candidates(
            candidates,
            selection_mode=selection_mode,
        )
        final_pred = selected["final_pred"]
        is_correct = answers_match(final_pred, true_a)
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

    return metrics, rows


def majority_vote(answers: Sequence[Optional[Union[str, int]]]) -> Optional[str]:
    """Return the most common non-null answer."""
    cleaned = [str(a) for a in answers if a is not None]
    if len(cleaned) == 0:
        return None
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


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
    system_prompt: str | None = None,
):
    """
    General evaluation function shared by:
    - baseline
    - self-consistency

    selection_mode:
    - 'majority_vote'
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

        # Shared candidate-generation path for all evaluation modes
        candidates = sample_candidate_texts(
            question=q,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )
        token_counts = sum_token_counts(candidates)

        total_input_tokens += token_counts["input_tokens"]
        total_new_tokens += token_counts["new_tokens"]
        total_tokens += token_counts["total_tokens"]

        selected = select_from_candidates(
            candidates,
            selection_mode=selection_mode,
        )
        final_pred = selected["final_pred"]
        is_correct = answers_match(final_pred, true_a)
        correct += int(is_correct)

        first_model_text = candidates[0]["text"] if candidates else ""
        if q in first_model_text:
            first_model_text = first_model_text.split(q, 1)[-1].strip()

        row = {
            "idx": int(i),
            "question": q,
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "n_samples": int(n_samples),
            "temperature": float(temperature),
            "model_output": first_model_text,
            "input_tokens": token_counts["input_tokens"],
            "new_tokens": token_counts["new_tokens"],
            "total_tokens": token_counts["total_tokens"],
            "selection_mode": selection_mode,
        }

        row["candidates"] = selected["candidate_preds"]

        rows.append(row)

    elapsed = time.time() - t0
    accuracy = correct / max(1, len(df))

    metrics = {
        "accuracy": accuracy,
        "num_questions": int(len(df)),
        "num_correct": int(correct),
        "n_samples": int(n_samples),
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "selection_mode": selection_mode,
        "system_prompt": system_prompt,
        "elapsed_sec": float(elapsed),
        "avg_input_tokens_per_question": total_input_tokens / max(1, len(df)),
        "avg_new_tokens_per_question": total_new_tokens / max(1, len(df)),
        "avg_total_tokens_per_question": total_tokens / max(1, len(df)),
    }

    return metrics, rows


def evaluate(
    eval_df,
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_questions: int = 100,
    max_new_tokens: int = 256,
    system_prompt: str | None = None,
):
    """
    Thin wrapper for baseline and self-consistency evaluation.
    """
    return evaluate_with_selection(
        eval_df=eval_df,
        model=model,
        tokenizer=tokenizer,
        n_samples=n_samples,
        temperature=temperature,
        max_questions=max_questions,
        max_new_tokens=max_new_tokens,
        selection_mode="majority_vote",
        system_prompt=system_prompt,
    )


# =============================================================================
# Difficulty bins
# =============================================================================

def assign_difficulty_bin(pass_rate):
    if pass_rate >= 0.8:
        return "easy"
    if pass_rate >= 0.3:
        return "medium"
    return "hard"


def assign_confidence_bin(confidence):
    """Map label-free answer agreement to easy/medium/hard bins."""
    if confidence >= 0.8:
        return "easy"
    if confidence >= 0.5:
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
        if answers_match(pred, true_answer):
            correct += 1

    return correct / max(1, total)


def estimate_answer_agreement_from_pool(candidates):
    """
    Estimate confidence without using labels.

    This is the fraction of valid candidate answers that agree with the modal
    answer. Low agreement means the model is uncertain across samples.
    """
    preds = [str(c.get("pred")) for c in candidates if c.get("pred") is not None]
    if not preds:
        return 0.0
    counts = Counter(preds)
    return counts.most_common(1)[0][1] / len(preds)


def build_difficulty_table_from_pool(generation_pool):
    """
    Build a difficulty table using a frozen proposal pool.
    This avoids regenerating candidates again.

    The pass-rate difficulty uses gold answers and is useful for analysis.
    The agreement-based difficulty uses only candidate agreement and can
    support a future routing policy without answer-label leakage.
    """
    rows = []

    for rec in generation_pool:
        idx = rec["idx"]
        q = rec["question"]
        true_a = rec["true_answer"]
        candidates = rec["candidates"]

        pass_rate = estimate_pass_rate_from_pool(candidates, true_a)
        difficulty = assign_difficulty_bin(pass_rate)
        answer_agreement = estimate_answer_agreement_from_pool(candidates)
        routing_difficulty = assign_confidence_bin(answer_agreement)

        rows.append({
            "idx": int(idx),
            "question": q,
            "true_answer": true_a,
            "pass_rate": float(pass_rate),
            "difficulty": difficulty,
            "answer_agreement": float(answer_agreement),
            "routing_difficulty": routing_difficulty,
            "n_samples_used": int(len(candidates)),
        })

    return pd.DataFrame(rows)


def attach_difficulty_columns(eval_df, difficulty_df):
    """Attach analysis and routing difficulty columns to the evaluated subset."""
    eval_with_diff = (
        eval_df.iloc[:len(difficulty_df)]
        .copy()
        .reset_index()
        .rename(columns={"index": "idx"})
    )
    return eval_with_diff.merge(
        difficulty_df[[
            "idx",
            "difficulty",
            "pass_rate",
            "answer_agreement",
            "routing_difficulty",
        ]],
        on="idx",
        how="left",
    )


def summarize_result_rows_by_difficulty(
    result_rows,
    categories: Sequence[str] = ("medium", "hard"),
    stage: str = "results",
    difficulty_col: str = "difficulty",
    correct_col: str = "correct",
):
    """Summarize total/correct/accuracy for each requested difficulty bin."""
    result_df = pd.DataFrame(result_rows)

    rows = []
    for category in categories:
        if len(result_df) and difficulty_col in result_df.columns:
            subset = result_df[result_df[difficulty_col] == category]
        else:
            subset = pd.DataFrame()

        total = int(len(subset))
        correct = int(subset[correct_col].sum()) if total and correct_col in subset.columns else 0
        accuracy = correct / total if total else 0.0
        rows.append(
            {
                "stage": stage,
                "difficulty": category,
                "total_questions": total,
                "num_correct": correct,
                "accuracy": accuracy,
                "accuracy_pct": 100 * accuracy,
            }
        )

    return pd.DataFrame(rows)


def summarize_difficulty_bin_performance(
    eval_with_diff,
    result_rows,
    categories: Sequence[str] = ("medium", "hard"),
    stage: str = "before_revisions",
):
    """
    Attach difficulty labels to result rows, then summarize bin performance.

    This is useful for reporting how the existing model predictions performed
    on the medium and hard bins before applying revision chains.
    """
    result_df = pd.DataFrame(result_rows)
    difficulty_cols = ["idx", "difficulty"]
    labeled_results = result_df.merge(
        eval_with_diff[difficulty_cols],
        on="idx",
        how="left",
    )
    return summarize_result_rows_by_difficulty(
        labeled_results,
        categories=categories,
        stage=stage,
    )


def _total_correct_summary_row(label: str, result_rows) -> dict:
    result_df = pd.DataFrame(result_rows)
    total = int(len(result_df))
    correct = int(result_df["correct"].sum()) if total else 0
    accuracy = correct / total if total else 0.0
    return {
        "stage": label,
        "difficulty": "all",
        "total_questions": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": 100 * accuracy,
    }


def _replace_revised_rows(base_rows, revised_rows_by_difficulty: dict):
    revised_by_idx = {
        row["idx"]: row
        for rows in revised_rows_by_difficulty.values()
        for row in rows
    }
    return [revised_by_idx.get(row["idx"], row) for row in base_rows]


def build_revision_before_after_summary(
    pre_revision_stats,
    base_result_rows,
    revision_rows_last: dict,
    revision_rows_majority: dict,
    categories: Sequence[str] = ("medium", "hard"),
):
    """
    Build the Part 4 before/after dataframe.

    The bin rows compare only revised difficulty categories. The all rows show
    the full experiment total after replacing revised rows and leaving all
    non-revised base answers unchanged.
    """
    revision_last_stats = summarize_result_rows_by_difficulty(
        [row for rows in revision_rows_last.values() for row in rows],
        categories=categories,
        stage="After revisions (last)",
    )

    revision_majority_stats = summarize_result_rows_by_difficulty(
        [row for rows in revision_rows_majority.values() for row in rows],
        categories=categories,
        stage="After revisions (majority vote)",
    )

    total_correct_stats = pd.DataFrame([
        _total_correct_summary_row("Before revisions", base_result_rows),
        _total_correct_summary_row(
            "After revisions (last)",
            _replace_revised_rows(base_result_rows, revision_rows_last),
        ),
        _total_correct_summary_row(
            "After revisions (majority vote)",
            _replace_revised_rows(base_result_rows, revision_rows_majority),
        ),
    ])

    return pd.concat(
        [pre_revision_stats, revision_last_stats, revision_majority_stats, total_correct_stats],
        ignore_index=True,
    )


# =============================================================================
# Sequential revisions (self-refinement)
# =============================================================================

def build_revision_prompt(
    question: str,
    previous_attempt_text: str,
    revision_prompt_template: str,
) -> str:
    """
    Build a revision prompt that asks the model to critique and improve
    its previous attempt.

    This is revision-specific logic only. It does not duplicate the base
    generation or extraction helpers.
    """
    # The notebook owns the prompt text; helpers only fill per-example fields.
    return revision_prompt_template.format(
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
    max_new_tokens: int = 256,
    base_system_prompt: str | None = None,
    revision_system_prompt: str | None = None,
    revision_prompt_template: str | None = None,
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
    if revision_prompt_template is None:
        raise ValueError("revision_prompt_template is required for revision chains")

    # Attempt 0: baseline
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

    # Revision attempts
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
    - last: choose the final revision
    - majority_vote: choose the most common predicted answer across attempts
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

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def build_revision_chain_pool(
    eval_df,
    model,
    tokenizer,
    r_steps: int = 3,
    base_temperature: float = 0.0,
    rev_temperature: float = 0.7,
    max_questions: int = 30,
    max_new_tokens: int = 256,
    base_system_prompt: str | None = None,
    revision_system_prompt: str | None = None,
    revision_prompt_template: str | None = None,
):
    """
    Generate and store one revision chain per question.

    This mirrors the frozen proposal pool pattern, but each
    record stores a sequential chain rather than independent full solutions.
    Selection rules can then be compared on the exact same generated attempts.
    """
    t0 = time.time()

    df = eval_df.copy()
    if max_questions is not None:
        df = df.iloc[:max_questions]

    pool = []
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    for i in tqdm(range(len(df)), desc=f"Build revision chains (R={r_steps})"):
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
            revision_system_prompt=revision_system_prompt,
            revision_prompt_template=revision_prompt_template,
        )
        token_counts = sum_token_counts(attempts)

        total_input_tokens += token_counts["input_tokens"]
        total_new_tokens += token_counts["new_tokens"]
        total_tokens += token_counts["total_tokens"]

        row = {
            "idx": int(df.iloc[i]["idx"]) if "idx" in df.columns else int(i),
            "question": q,
            "true_answer": true_a,
            "attempts": attempts,
        }

        if "difficulty" in df.columns:
            row["difficulty"] = df.iloc[i]["difficulty"]
        if "pass_rate" in df.columns:
            row["pass_rate"] = df.iloc[i]["pass_rate"]

        pool.append(row)

    elapsed = time.time() - t0
    n_questions = max(1, len(df))

    meta = {
        "num_questions": int(len(df)),
        "r_steps": int(r_steps),
        "base_temperature": float(base_temperature),
        "rev_temperature": float(rev_temperature),
        "max_new_tokens": int(max_new_tokens),
        "base_system_prompt": base_system_prompt,
        "revision_system_prompt": revision_system_prompt,
        "revision_prompt_template": revision_prompt_template,
        "elapsed_sec": float(elapsed),
        "avg_input_tokens_per_question": total_input_tokens / n_questions,
        "avg_new_tokens_per_question": total_new_tokens / n_questions,
        "avg_total_tokens_per_question": total_tokens / n_questions,
    }

    return pool, meta


def evaluate_from_revision_chain_pool(
    revision_chain_pool,
    pool_meta: Optional[dict] = None,
    selection_mode: str = "last",
):
    """
    Score a selection rule on a frozen revision-chain pool.

    No model calls happen here. This isolates the selection rule while keeping
    generation constant across last-revision and majority-vote modes.
    """
    t0 = time.time()
    pool_meta = pool_meta or {}

    rows = []
    correct = 0
    total_input_tokens = 0
    total_new_tokens = 0
    total_tokens = 0

    for rec in tqdm(revision_chain_pool, desc=f"Eval revision pool: {selection_mode}"):
        attempts = rec["attempts"]
        true_a = rec.get("true_answer")

        selected = select_revision_attempt(
            attempts=attempts,
            selection_mode=selection_mode,
        )

        final_pred = selected.get("pred")
        is_correct = answers_match(final_pred, true_a)
        correct += int(is_correct)

        token_counts = sum_token_counts(attempts)
        total_input_tokens += token_counts["input_tokens"]
        total_new_tokens += token_counts["new_tokens"]
        total_tokens += token_counts["total_tokens"]

        row = {
            "idx": int(rec["idx"]),
            "question": rec["question"],
            "true_answer": true_a,
            "final_pred": final_pred,
            "correct": bool(is_correct),
            "r_steps": int(pool_meta.get("r_steps", max(0, len(attempts) - 1))),
            "base_temperature": float(pool_meta.get("base_temperature", 0.0)),
            "rev_temperature": float(pool_meta.get("rev_temperature", 0.7)),
            "selection_mode": selection_mode,
            "selected_attempt": selected.get("attempt"),
            "attempts": attempts,
        }

        if "difficulty" in rec:
            row["difficulty"] = rec["difficulty"]
        if "pass_rate" in rec:
            row["pass_rate"] = rec["pass_rate"]
        rows.append(row)

    selection_elapsed = time.time() - t0
    n_questions = max(1, len(revision_chain_pool))
    accuracy = correct / n_questions

    metrics = {
        "method": "sequential_revisions",
        "accuracy": float(accuracy),
        "num_questions": int(len(revision_chain_pool)),
        "num_correct": int(correct),
        "r_steps": int(pool_meta.get("r_steps", 0)),
        "base_temperature": float(pool_meta.get("base_temperature", 0.0)),
        "rev_temperature": float(pool_meta.get("rev_temperature", 0.7)),
        "selection_mode": selection_mode,
        "elapsed_sec": float(pool_meta.get("elapsed_sec", 0.0) + selection_elapsed),
        "generation_elapsed_sec": float(pool_meta.get("elapsed_sec", 0.0)),
        "selection_elapsed_sec": float(selection_elapsed),
        "avg_input_tokens_per_question": total_input_tokens / n_questions,
        "avg_new_tokens_per_question": total_new_tokens / n_questions,
        "avg_total_tokens_per_question": total_tokens / n_questions,
    }

    return metrics, rows



def build_experiment_payload(
    experiment_config: dict,
    baseline_metrics: dict,
    proposal_pool_meta: dict,
    baseline_prompt: str,
    revision_system_prompt: str,
    revision_prompt_template: str,
    preview_token_stats,
    preview_token_df,
    baseline_results,
    sc_metrics: dict,
    sc_results,
    difficulty_df,
    revision_pool_meta: dict,
    revision_chain_pool,
    rev_last_metrics: dict,
    rev_mv_metrics: dict,
) -> dict:
    """Build the full experiment artifact payload saved by the notebook."""
    return {
        "config": {
            **experiment_config,
            "baseline_max_questions": baseline_metrics.get("num_questions"),
            "pool_num_questions": proposal_pool_meta.get("num_questions"),
            "pool_n_samples": proposal_pool_meta.get("n_samples"),
            "pool_temperature": proposal_pool_meta.get("temperature"),
        },
        "prompts": {
            "BASELINE_PROMPT": baseline_prompt,
            "revision_system_prompt": revision_system_prompt,
            "revision_prompt_template": revision_prompt_template,
        },
        "preview_token_stats": preview_token_stats.to_dict(orient="records"),
        "preview_token_rows": preview_token_df.to_dict(orient="records"),
        "baseline_metrics": baseline_metrics,
        "baseline_rows": baseline_results,
        "self_consistency_metrics": sc_metrics,
        "self_consistency_rows": sc_results,
        "difficulty_bins": difficulty_df.to_dict(orient="records"),
        "revision_chain_pool_meta": revision_pool_meta,
        "revision_chain_pool": revision_chain_pool,
        "revision_metrics": {
            "last": rev_last_metrics,
            "majority_vote": rev_mv_metrics,
        },
    }


def _compact_metric_row(name: str, metrics: dict) -> dict:
    row = dict(metrics)
    row["method"] = name
    if "avg_total_tokens_per_question" in row:
        row["avg_tokens"] = row["avg_total_tokens_per_question"]
    elif "avg_new_tokens_per_question" in row:
        row["avg_tokens"] = row["avg_new_tokens_per_question"]
    else:
        row["avg_tokens"] = None
    return row


def build_metrics_table(named_metrics: dict):
    """Create the compact final metrics dataframe shown at the end."""
    rows = [_compact_metric_row(name, metrics) for name, metrics in named_metrics.items()]
    return pd.DataFrame(rows)


def metrics_summary_view(metrics_df):
    """Return the notebook-facing subset of final metric columns."""
    summary_columns = [
        "method",
        "accuracy",
        "num_correct",
        "num_questions",
        "avg_tokens",
        "elapsed_sec",
        "selection_mode",
    ]
    summary_columns = [col for col in summary_columns if col in metrics_df.columns]
    return metrics_df[summary_columns]


def plot_accuracy_vs_compute(metrics_df) -> None:
    """Plot accuracy against average token cost for completed metric rows."""
    import matplotlib.pyplot as plt

    plot_df = metrics_df.dropna(subset=["accuracy", "avg_tokens"])

    if len(plot_df):
        plt.figure(figsize=(8, 5))
        plt.scatter(plot_df["avg_tokens"], plot_df["accuracy"])
        for _, row in plot_df.iterrows():
            plt.annotate(row["method"], (row["avg_tokens"], row["accuracy"]), fontsize=8)
        plt.xlabel("Average tokens per question")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Test-Time Compute")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No metric rows have both accuracy and token counts yet.")


def write_results_summary(
    path: str,
    metrics_df,
) -> str:
    """Write the compact summary artifact used for quick inspection."""
    out = {
        "metrics_table": metrics_df.to_dict(orient="records"),
    }
    return write_json_artifact(out, path)


def write_json_artifact(payload: dict, path: str) -> str:
    """Write a JSON artifact and print its path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")
    return path
