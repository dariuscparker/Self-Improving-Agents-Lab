from __future__ import annotations

import ast
import json
import math
import re
import signal
import textwrap
import traceback
import torch
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime in notebooks
    pd = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - handled at runtime in notebooks
    def tqdm(iterable, **kwargs):
        return iterable


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


class TimeoutErrorDuringExec(Exception):
    """Raised when a generated candidate exceeds the execution time limit."""


@contextmanager
def time_limit(seconds: int):
    """Apply a SIGALRM-based timeout while executing generated code."""

    def handler(signum, frame):
        raise TimeoutErrorDuringExec(f"Timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def strip_markdown_fences(text: str) -> str:
    """Extract Python from a fenced markdown block when the model emits one."""
    text = (text or "").replace("\r\n", "\n")
    match = CODE_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    if text.lower().startswith("python\n"):
        text = text.split("\n", 1)[1]
    return text.strip("\n")


def _compiles_as_python(source: str) -> bool:
    """Return True when `source` is syntactically valid Python."""
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError:
        return False
    return True


def _indent_function_body(body: str) -> str:
    """Indent non-empty lines to one function-body level."""
    return textwrap.indent(body, "    ", predicate=lambda line: line.strip() != "")


def _strip_body_indent(body: str, amount: int) -> str:
    """Shift a body block left by `amount` spaces where possible."""
    lines = body.splitlines()
    repaired_lines: List[str] = []
    for line in lines:
        if not line.strip():
            repaired_lines.append("")
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent >= amount:
            repaired_lines.append(line[amount:])
        else:
            repaired_lines.append(line.lstrip(" "))

    return "\n".join(repaired_lines)


def assemble_candidate_source(prompt: str, completion: str, entry_point: str) -> str:
    prompt_text = prompt or ""
    cleaned = strip_markdown_fences(completion).expandtabs(4).rstrip()
    cleaned = textwrap.dedent(cleaned)

    if not cleaned:
        return prompt_text + "\n"

    # If model returned full function
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", cleaned, flags=re.MULTILINE):
        return cleaned + "\n"

    # Ensure prompt ends cleanly
    if not prompt_text.endswith("\n"):
        prompt_text += "\n"

    variants = [
        _indent_function_body(cleaned),
    ]

    lines = [line for line in cleaned.splitlines() if line.strip()]
    if lines:
        indents = sorted(
            {
                len(line) - len(line.lstrip(" "))
                for line in lines
                if len(line) - len(line.lstrip(" ")) > 0
            },
            reverse=True,
        )
        for indent in indents:
            repaired = _strip_body_indent(cleaned, indent)
            body = _indent_function_body(repaired)
            if body not in variants:
                variants.append(body)

    for body in variants:
        source = prompt_text + body + "\n"
        if _compiles_as_python(source):
            return source

    return prompt_text + variants[-1] + "\n"


def build_test_program(problem: Dict[str, Any], completion: str) -> str:
    """Combine candidate code with the hidden HumanEval checks."""
    return build_test_program_with_test_code(problem, completion, problem["test"])


def build_test_program_with_test_code(
    problem: Dict[str, Any],
    completion: str,
    test_code: str,
) -> str:
    """Combine candidate code with a specific HumanEval test program."""
    source = assemble_candidate_source(
        prompt=problem["prompt"],
        completion=completion,
        entry_point=problem["entry_point"],
    )

    parts = [
        source.rstrip(),
        "",
        test_code.rstrip(),
        "",
        f"check({problem['entry_point']})",
        "",
    ]
    return "\n".join(parts)


def evaluate_candidate(
    problem: Dict[str, Any],
    completion: str,
    timeout_seconds: int = 5,
    test_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one generated completion against the HumanEval tests."""
    program = build_test_program_with_test_code(
        problem,
        completion,
        test_code if test_code is not None else problem["test"],
    )

    result = {
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "passed": False,
        "status": "unknown",
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "program": program,
        "raw_completion": completion,
    }

    try:
        compiled = compile(program, f"<{problem['task_id']}>", "exec")
    except SyntaxError as exc:
        result.update(
            status="syntax_error",
            error_type="SyntaxError",
            error_message=str(exc),
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        return result

    namespace: Dict[str, Any] = {}
    try:
        with time_limit(timeout_seconds):
            exec(compiled, namespace, namespace)
    except TimeoutErrorDuringExec as exc:
        result.update(
            status="timeout",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except AssertionError as exc:
        result.update(
            status="failed_tests",
            error_type="AssertionError",
            error_message=str(exc) if str(exc) else "One or more tests failed",
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="runtime_error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
    else:
        result.update(status="passed", passed=True)

    return result


def debug_example(
    problem: Dict[str, Any],
    build_messages_fn: Callable[[Dict[str, Any]], List[Dict[str, str]]],
    generate_fn: Callable[..., List[str]],
    tokenizer,
    max_new_tokens: int = 512,
    timeout_seconds: int = 5,
    temperature: float = 0.0,
    top_p: float = 0.95,
    print_trace: bool = True,
) -> Dict[str, Any]:
    """Run a single end-to-end HumanEval debug example and return a structured result."""
    messages = build_messages_fn(problem)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    completions = generate_fn(
        problem,
        num_samples=1,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
    )
    raw_completion = completions[0] if completions else ""

    assembled_program = build_test_program(problem, raw_completion)
    evaluation = evaluate_candidate(
        problem=problem,
        completion=raw_completion,
        timeout_seconds=timeout_seconds,
    )

    debug_info = {
        "problem": problem,
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "messages": messages,
        "prompt_text": prompt_text,
        "raw_completion": raw_completion,
        "assembled_program": assembled_program,
        "evaluation": evaluation,
    }

    if print_trace:
        print("TASK ID:", debug_info["task_id"])
        print("ENTRY POINT:", debug_info["entry_point"])
        print("\n" + "=" * 80)
        print("ORIGINAL HUMANEVAL PROMPT")
        print("=" * 80)
        print(problem["prompt"])
        print("\n" + "=" * 80)
        print("CHAT PROMPT SENT TO MODEL")
        print("=" * 80)
        print(prompt_text)
        print("\n" + "=" * 80)
        print("RAW MODEL COMPLETION")
        print("=" * 80)
        print(raw_completion)
        print("\n" + "=" * 80)
        print("ASSEMBLED PROGRAM (PROMPT + COMPLETION + TESTS)")
        print("=" * 80)
        print(assembled_program)
        print("\n" + "=" * 80)
        print("EVALUATION RESULT")
        print("=" * 80)
        for key in ["passed", "status", "error_type", "error_message"]:
            print(f"{key}: {evaluation.get(key)}")

    return debug_info


def _to_source(node: ast.AST) -> str:
    """Convert an AST node back into Python source."""
    return ast.unparse(node)


def _split_check_into_chunks(check_node: ast.FunctionDef) -> List[List[ast.stmt]]:
    """
    Split a HumanEval `check(candidate)` body into test chunks.

    Each chunk ends at an `assert` statement and includes any setup statements
    immediately before that assert so the chunk can run independently.
    """
    chunks: List[List[ast.stmt]] = []
    buffer: List[ast.stmt] = []

    for stmt in check_node.body:
        buffer.append(stmt)
        if isinstance(stmt, ast.Assert):
            chunks.append(buffer)
            buffer = []

    if buffer:
        if chunks:
            chunks[-1].extend(buffer)
        else:
            chunks.append(buffer)

    if not chunks:
        return [[ast.Pass()]]
    return chunks


def split_humaneval_test_cases(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a HumanEval test string into reusable support code plus test chunks.

    Returns a structure that can be used to build public and hidden test sets.
    """
    module = ast.parse(problem["test"])
    support_nodes: List[ast.stmt] = []
    check_node: Optional[ast.FunctionDef] = None

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_node = node
        else:
            support_nodes.append(node)

    if check_node is None:
        raise ValueError(f"No check(candidate) function found for {problem['task_id']}")

    chunks = _split_check_into_chunks(check_node)
    return {
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "support_code": "\n\n".join(_to_source(node) for node in support_nodes).strip(),
        "chunk_codes": [
            "\n".join(_to_source(stmt) for stmt in chunk).strip()
            for chunk in chunks
        ],
        "num_chunks": len(chunks),
    }


def build_split_test_code(
    support_code: str,
    chunk_codes: Sequence[str],
) -> str:
    """Rebuild a `check(candidate)` function from a subset of test chunks."""
    body_lines: List[str] = []
    for chunk in chunk_codes:
        for line in chunk.splitlines():
            body_lines.append(f"    {line}")

    if not body_lines:
        body_lines = ["    pass"]

    parts = []
    if support_code:
        parts.append(support_code.rstrip())
    parts.append("def check(candidate):")
    parts.append("\n".join(body_lines))
    return "\n\n".join(parts) + "\n"


def make_public_private_test_split(
    problem: Dict[str, Any],
    public_fraction: float = 0.4,
    min_public_cases: int = 1,
) -> Dict[str, Any]:
    """
    Create a deterministic public/private split for one HumanEval problem.

    The first chunked assertions become the public verifier set; the remainder
    stay private for final evaluation.
    """
    parsed = split_humaneval_test_cases(problem)
    num_chunks = parsed["num_chunks"]
    if num_chunks == 1:
        # A single chunk is the entire hidden test suite, so exposing it would
        # leak the evaluator and invalidate the verifier experiment.
        public_count = 0
    else:
        public_count = max(min_public_cases, int(math.ceil(num_chunks * public_fraction)))
        public_count = min(public_count, num_chunks - 1)

    public_chunks = parsed["chunk_codes"][:public_count]
    private_chunks = parsed["chunk_codes"][public_count:]

    return {
        **parsed,
        "public_count": public_count,
        "private_count": len(private_chunks),
        "public_test_code": build_split_test_code(parsed["support_code"], public_chunks),
        "private_test_code": build_split_test_code(parsed["support_code"], private_chunks),
    }


def score_candidate_on_public_tests(
    problem: Dict[str, Any],
    completion: str,
    split_info: Dict[str, Any],
    timeout_seconds: int = 5,
) -> Dict[str, Any]:
    """
    Score one candidate using only public test chunks.

    The score is the fraction of public chunks passed individually, which gives
    a weak but deployable verifier signal.
    """
    public_chunk_codes = split_info["chunk_codes"][: split_info["public_count"]]
    per_chunk_results: List[Dict[str, Any]] = []

    for chunk_index, chunk_code in enumerate(public_chunk_codes):
        chunk_test_code = build_split_test_code(split_info["support_code"], [chunk_code])
        chunk_result = evaluate_candidate(
            problem=problem,
            completion=completion,
            timeout_seconds=timeout_seconds,
            test_code=chunk_test_code,
        )
        per_chunk_results.append(
            {
                "chunk_index": chunk_index,
                "passed": chunk_result["passed"],
                "status": chunk_result["status"],
                "error_type": chunk_result["error_type"],
            }
        )

    passed_count = sum(1 for row in per_chunk_results if row["passed"])
    total_public = len(per_chunk_results)
    return {
        "public_pass_count": passed_count,
        "public_total": total_public,
        "public_score": passed_count / total_public if total_public else 0.0,
        "public_details": per_chunk_results,
    }


def summarize_statuses(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count execution statuses for a result collection."""
    counts: Dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _require_pandas() -> None:
    """Raise a helpful error if pandas is not installed."""
    if pd is None:
        raise ImportError("pandas is required for DataFrame-based HumanEval utilities.")


def baseline_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate pass@1 and error rates from one-candidate-per-task evaluation."""
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    syntax_errors = sum(1 for row in rows if row["status"] == "syntax_error")
    runtime_failures = sum(
        1 for row in rows if row["status"] in {"runtime_error", "timeout", "failed_tests"}
    )

    return {
        "method": "baseline_pass@1",
        "num_tasks": total,
        "num_passed": passed,
        "pass@1": passed / total if total else 0.0,
        "syntax_errors": syntax_errors,
        "syntax_error_rate": syntax_errors / total if total else 0.0,
        "runtime_failures": runtime_failures,
        "runtime_failure_rate": runtime_failures / total if total else 0.0,
        "status_counts": summarize_statuses(rows),
    }


def run_baseline(
    problems_df: pd.DataFrame,
    generate_fn: Callable[..., List[str]],
    max_tasks: Optional[int] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    timeout_seconds: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate one solution per problem and evaluate it with HumanEval tests."""
    _require_pandas()
    if max_tasks is not None:
        problems_df = problems_df.iloc[:max_tasks].copy()

    rows: List[Dict[str, Any]] = []

    for _, problem in tqdm(problems_df.iterrows(), total=len(problems_df), desc="Baseline pass@1"):
        problem_dict = problem.to_dict()
        completions = generate_fn(
            problem_dict,
            num_samples=1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        raw_completion = completions[0] if completions else ""
        result = evaluate_candidate(
            problem=problem_dict,
            completion=raw_completion,
            timeout_seconds=timeout_seconds,
        )
        rows.append(result)

    return baseline_metrics(rows), rows


def build_candidate_pool(
    problems_df: pd.DataFrame,
    generate_fn: Callable[..., List[str]],
    num_candidates: int = 5,
    max_tasks: Optional[int] = None,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    timeout_seconds: int = 5,
) -> pd.DataFrame:
    """Generate multiple candidates per problem and evaluate each one."""
    _require_pandas()
    if max_tasks is not None:
        problems_df = problems_df.iloc[:max_tasks].copy()

    rows: List[Dict[str, Any]] = []

    for _, problem in tqdm(
        problems_df.iterrows(),
        total=len(problems_df),
        desc=f"Generating {num_candidates} candidates",
    ):
        problem_dict = problem.to_dict()
        completions = generate_fn(
            problem_dict,
            num_samples=num_candidates,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        for sample_index, raw_completion in enumerate(completions):
            result = evaluate_candidate(
                problem=problem_dict,
                completion=raw_completion,
                timeout_seconds=timeout_seconds,
            )
            result["sample_index"] = sample_index
            rows.append(result)

    pool_df = pd.DataFrame(rows).sort_values(["task_id", "sample_index"]).reset_index(drop=True)
    return pool_df


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    HumanEval-style pass@k estimator from OpenAI's evaluation code.

    Returns the probability that at least one of k samples is correct when
    drawing without replacement from num_samples samples containing num_correct
    correct programs.
    """
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0

    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - (k / value)
    return 1.0 - product


def pass_at_k_table(pool_df: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """Compute empirical and estimated pass@k values from a frozen candidate pool."""
    _require_pandas()
    task_groups = {
        task_id: group.sort_values("sample_index").reset_index(drop=True)
        for task_id, group in pool_df.groupby("task_id", sort=True)
    }

    rows: List[Dict[str, Any]] = []
    for k in k_values:
        empirical_hits = []
        estimated_values = []

        for group in task_groups.values():
            capped = group.iloc[:k]
            empirical_hits.append(bool(capped["passed"].any()))

            total_samples = len(group)
            num_correct = int(group["passed"].sum())
            estimated_values.append(estimate_pass_at_k(total_samples, num_correct, k))

        rows.append(
            {
                "k": k,
                "num_tasks": len(task_groups),
                "empirical_pass@k": sum(empirical_hits) / len(empirical_hits) if empirical_hits else 0.0,
                "estimated_pass@k": sum(estimated_values) / len(estimated_values) if estimated_values else 0.0,
            }
        )

    return pd.DataFrame(rows)


def oracle_best_of_n(pool_df: pd.DataFrame, n: int) -> Dict[str, Any]:
    """
    Compute an oracle best-of-N upper bound by selecting any passing candidate.

    This is not deployable because it assumes access to the full hidden tests
    for every candidate at selection time.
    """
    task_groups = {
        task_id: group.sort_values("sample_index").reset_index(drop=True).iloc[:n]
        for task_id, group in pool_df.groupby("task_id", sort=True)
    }
    hits = [bool(group["passed"].any()) for group in task_groups.values()]

    return {
        "method": f"oracle_best_of_{n}",
        "num_tasks": len(task_groups),
        "oracle_accuracy": sum(hits) / len(hits) if hits else 0.0,
        "deployable": False,
        "reason": (
            "Selection uses hidden evaluation tests to identify the winning candidate, "
            "which is only valid as an upper bound."
        ),
    }


def evaluate_public_test_verifier(
    problems_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    public_fraction: float = 0.4,
    timeout_seconds: int = 5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Select one candidate per task using only public test chunks, then score it
    on the original hidden HumanEval outcome stored in `pool_df`.
    """
    _require_pandas()
    task_lookup = {
        row["task_id"]: row
        for _, row in problems_df.iterrows()
    }

    selected_rows: List[Dict[str, Any]] = []

    for task_id, group in pool_df.groupby("task_id", sort=True):
        problem = task_lookup[task_id].to_dict() if hasattr(task_lookup[task_id], "to_dict") else dict(task_lookup[task_id])
        split_info = make_public_private_test_split(problem, public_fraction=public_fraction)

        candidate_rows: List[Dict[str, Any]] = []
        for _, candidate in group.sort_values("sample_index").iterrows():
            candidate_dict = candidate.to_dict()
            public_eval = score_candidate_on_public_tests(
                problem=problem,
                completion=candidate_dict["raw_completion"],
                split_info=split_info,
                timeout_seconds=timeout_seconds,
            )
            candidate_rows.append({**candidate_dict, **public_eval})

        ranked = sorted(
            candidate_rows,
            key=lambda row: (
                row["public_score"],
                row["public_pass_count"],
                -row["sample_index"],
            ),
            reverse=True,
        )
        chosen = ranked[0]
        selected_rows.append(
            {
                "task_id": task_id,
                "selected_sample_index": chosen["sample_index"],
                "selected_passed_full_tests": chosen["passed"],
                "selected_status": chosen["status"],
                "public_score": chosen["public_score"],
                "public_pass_count": chosen["public_pass_count"],
                "public_total": chosen["public_total"],
                "oracle_available": bool(group["passed"].any()),
                "num_candidates": len(group),
                "num_public_cases": split_info["public_count"],
                "num_total_cases": split_info["num_chunks"],
            }
        )

    selections_df = pd.DataFrame(selected_rows).sort_values("task_id").reset_index(drop=True)
    accuracy = selections_df["selected_passed_full_tests"].mean() if len(selections_df) else 0.0
    metrics = {
        "method": "public_test_verifier",
        "num_tasks": len(selections_df),
        "accuracy": float(accuracy),
        "avg_public_score": float(selections_df["public_score"].mean()) if len(selections_df) else 0.0,
        "avg_public_cases": float(selections_df["num_public_cases"].mean()) if len(selections_df) else 0.0,
    }
    return metrics, selections_df


def evaluate_llm_judge_selector(
    problems_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    judge_fn: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Select one candidate per task using a judge model, then score it on the full
    HumanEval outcome already stored in `pool_df`.
    """
    _require_pandas()
    task_lookup = {
        row["task_id"]: row
        for _, row in problems_df.iterrows()
    }

    selected_rows: List[Dict[str, Any]] = []

    for task_id, group in pool_df.groupby("task_id", sort=True):
        problem = task_lookup[task_id].to_dict() if hasattr(task_lookup[task_id], "to_dict") else dict(task_lookup[task_id])
        candidates = [
            row.to_dict()
            for _, row in group.sort_values("sample_index").iterrows()
        ]
        judge_output = judge_fn(problem, candidates)
        selected_index = int(judge_output.get("selected_sample_index", 0))

        matching = [row for row in candidates if int(row["sample_index"]) == selected_index]
        chosen = matching[0] if matching else candidates[0]

        selected_rows.append(
            {
                "task_id": task_id,
                "selected_sample_index": int(chosen["sample_index"]),
                "selected_passed_full_tests": bool(chosen["passed"]),
                "selected_status": chosen["status"],
                "oracle_available": bool(group["passed"].any()),
                "judge_raw_output": judge_output.get("judge_raw_output"),
                "judge_reason": judge_output.get("judge_reason"),
            }
        )

    selections_df = pd.DataFrame(selected_rows).sort_values("task_id").reset_index(drop=True)
    accuracy = selections_df["selected_passed_full_tests"].mean() if len(selections_df) else 0.0
    metrics = {
        "method": "llm_judge",
        "num_tasks": len(selections_df),
        "accuracy": float(accuracy),
    }
    return metrics, selections_df


def truncate_for_judge(text: str, max_chars: int = 1200) -> str:
    """Trim candidate code so judge prompts stay compact."""
    cleaned = strip_markdown_fences(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "\n# ... truncated for judge ..."


def parse_judge_response(
    raw_text: str,
    valid_indices: Sequence[int],
) -> Tuple[int, Optional[str]]:
    """Parse a judge model response into a selected candidate index."""
    raw_text = (raw_text or "").strip()

    def _validated_index(value: int) -> Optional[int]:
        return value if value in valid_indices else None

    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            selected_index = int(parsed.get("selected_sample_index", 0))
            validated = _validated_index(selected_index)
            if validated is not None:
                return validated, parsed.get("judge_reason") or parsed.get("reason")
        except Exception:  # noqa: BLE001
            pass

    key_match = re.search(
        r"(?:selected_sample_index|selected\s+candidate(?:\s+index)?)\s*[:=]?\s*(\d+)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if key_match:
        validated = _validated_index(int(key_match.group(1)))
        if validated is not None:
            return validated, None

    selection_patterns = [
        r"\b(?:choose|chose|pick|picked|select|selected)\s+(?:candidate\s+)?(\d+)\b",
        r"\bcandidate\s+(\d+)\s+(?:is|looks|seems|appears)\s+(?:the\s+)?(?:best|better|strongest|correct|preferred)\b",
        r"\b(\d+)\s+(?:is|looks|seems|appears)\s+(?:the\s+)?(?:best|better|strongest|correct|preferred)\b",
        r"\bcandidate\s+(\d+)\s+(?:wins|win|beats|beat)\b",
        r"\b(\d+)\s+(?:wins|win|beats|beat)\b",
        r"^\s*(?:candidate\s+)?(\d+)\s*$",
    ]
    for pattern in selection_patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            validated = _validated_index(int(match.group(1)))
            if validated is not None:
                return validated, None

    fallback_index = min(valid_indices) if valid_indices else 0
    return fallback_index, "Fallback to the first candidate because the judge output was not parseable."


def build_judge_prompt(
    problem: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    max_candidate_chars: int = 1200,
) -> str:
    """Build the user prompt for the LLM-as-a-judge selector."""
    candidate_blocks = []
    for candidate in candidates:
        sample_index = int(candidate["sample_index"])
        candidate_blocks.append(
            f"Candidate {sample_index}:\n```python\n"
            f"{truncate_for_judge(candidate['raw_completion'], max_chars=max_candidate_chars)}\n```"
        )

    return (
        "You are judging candidate Python completions for a HumanEval task. "
        "Choose the candidate most likely to pass hidden tests. "
        "Consider correctness, edge cases, types, and whether the code fully matches the prompt. "
        "Do not run the code. Respond with compact JSON only.\n\n"
        f"Task prompt:\n{problem['prompt']}\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nReturn JSON exactly in this form: "
        + '{"selected_sample_index": <integer>, "judge_reason": "short reason"}'
    )


def make_llm_judge_fn(
    model,
    tokenizer,
    max_candidate_chars: int = 1200,
    max_new_tokens: int = 96,
):
    """Create a judge function compatible with `evaluate_llm_judge_selector`."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - notebook runtime concern
        raise ImportError("torch is required for LLM judge evaluation.") from exc

    def judge_fn(problem: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_indices = [int(candidate["sample_index"]) for candidate in candidates]
        judge_user_prompt = build_judge_prompt(
            problem,
            candidates,
            max_candidate_chars=max_candidate_chars,
        )

        messages = [
            {"role": "system", "content": "You are a precise code judge. Return JSON only."},
            {"role": "user", "content": judge_user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        raw_output = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        selected_index, judge_reason = parse_judge_response(raw_output, valid_indices)

        return {
            "selected_sample_index": selected_index,
            "judge_reason": judge_reason,
            "judge_raw_output": raw_output,
        }

    return judge_fn


def build_results_table(
    baseline: Dict[str, Any],
    passk_df: pd.DataFrame,
    oracle: Dict[str, Any],
    public_metrics: Optional[Dict[str, Any]] = None,
    judge_metrics: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Assemble the assignment's required top-line comparison table."""
    _require_pandas()
    rows = [
        {
            "method": "Baseline",
            "accuracy": baseline.get("pass@1"),
            "notes": "Single generated completion",
        }
    ]

    for _, row in passk_df.iterrows():
        k_value = int(row["k"])
        if k_value == 1:
            notes = (
                "First sampled candidate from the frozen pool; compare carefully if the pool "
                "used different generation settings than the baseline"
            )
        else:
            notes = "At least one passing candidate among first k samples"
        rows.append(
            {
                "method": f"Pass@{k_value}",
                "accuracy": float(row["empirical_pass@k"]),
                "notes": notes,
            }
        )

    rows.append(
        {
            "method": oracle["method"],
            "accuracy": oracle["oracle_accuracy"],
            "notes": oracle["reason"],
        }
    )

    if public_metrics is not None:
        rows.append(
            {
                "method": "Public-test verifier",
                "accuracy": public_metrics["accuracy"],
                "notes": "Candidate chosen using only partial public tests",
            }
        )

    if judge_metrics is not None:
        rows.append(
            {
                "method": "LLM judge",
                "accuracy": judge_metrics["accuracy"],
                "notes": "Candidate chosen by the judge model without hidden tests",
            }
        )

    return pd.DataFrame(rows)


def build_failure_analysis_table(
    pool_df: pd.DataFrame,
    public_selections_df: pd.DataFrame,
    judge_selections_df: Optional[pd.DataFrame] = None,
    max_cases: int = 5,
) -> pd.DataFrame:
    """
    Build a compact table of interesting verification outcomes.

    Prioritizes:
    1. public verifier rescues a task where sample 0 failed
    2. public verifier misses a task where oracle had a passing candidate
    3. judge misses a task where oracle had a passing candidate
    """
    _require_pandas()
    first_sample = (
        pool_df.sort_values(["task_id", "sample_index"])
        .groupby("task_id", as_index=False)
        .first()[["task_id", "passed", "sample_index"]]
        .rename(columns={"passed": "first_sample_passed", "sample_index": "first_sample_index"})
    )
    oracle_df = (
        pool_df.groupby("task_id", as_index=False)["passed"]
        .max()
        .rename(columns={"passed": "oracle_available"})
    )

    merged = public_selections_df.merge(first_sample, on="task_id", how="left").merge(
        oracle_df,
        on="task_id",
        how="left",
        suffixes=("", "_oracle"),
    )

    if judge_selections_df is not None:
        merged = merged.merge(
            judge_selections_df[["task_id", "selected_passed_full_tests"]].rename(
                columns={"selected_passed_full_tests": "judge_passed"}
            ),
            on="task_id",
            how="left",
        )

    def classify(row: pd.Series) -> str:
        if (not row["first_sample_passed"]) and row["selected_passed_full_tests"]:
            return "public verifier helped"
        if row["oracle_available"] and (not row["selected_passed_full_tests"]):
            return "public verifier missed available solution"
        if bool(row.get("judge_passed", True)) is False and row["oracle_available"]:
            return "judge missed available solution"
        return "reference case"

    merged["analysis_label"] = merged.apply(classify, axis=1)
    priority = {
        "public verifier helped": 0,
        "public verifier missed available solution": 1,
        "judge missed available solution": 2,
        "reference case": 3,
    }
    merged["priority"] = merged["analysis_label"].map(priority).fillna(99)

    columns = [
        "task_id",
        "analysis_label",
        "first_sample_passed",
        "selected_passed_full_tests",
        "oracle_available",
        "public_score",
        "selected_sample_index",
    ]
    if "judge_passed" in merged.columns:
        columns.append("judge_passed")

    return (
        merged.sort_values(["priority", "task_id"])
        .head(max_cases)[columns]
        .reset_index(drop=True)
    )


def collect_case_artifacts(
    task_id: str,
    problems_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    public_selections_df: pd.DataFrame,
    judge_selections_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Gather all tables needed to inspect one failure-analysis case."""
    _require_pandas()
    problem = problems_df.loc[problems_df["task_id"] == task_id].iloc[0]
    candidate_view = pool_df.loc[
        pool_df["task_id"] == task_id,
        ["sample_index", "passed", "status", "error_type", "error_message"],
    ].copy()
    public_view = public_selections_df.loc[public_selections_df["task_id"] == task_id].copy()
    judge_view = None
    if judge_selections_df is not None and len(judge_selections_df):
        judge_view = judge_selections_df.loc[judge_selections_df["task_id"] == task_id].copy()

    return {
        "task_id": task_id,
        "prompt": problem["prompt"],
        "candidate_view": candidate_view,
        "public_view": public_view,
        "judge_view": judge_view,
    }


def summary_table(
    baseline: Dict[str, Any],
    passk_df: pd.DataFrame,
    oracle: Dict[str, Any],
) -> pd.DataFrame:
    """Create a compact comparison table for notebook reporting."""
    _require_pandas()
    rows = [
        {
            "method": "Baseline",
            "metric": "pass@1",
            "value": baseline["pass@1"],
            "notes": "One completion per task",
        }
    ]

    for _, row in passk_df.iterrows():
        k_value = int(row["k"])
        if k_value == 1:
            notes = (
                "First sampled candidate from the frozen pool; not identical to the baseline "
                "if temperatures differ"
            )
        else:
            notes = "At least one passing candidate among first k samples"
        rows.append(
            {
                "method": f"Pass@{k_value}",
                "metric": "empirical_pass@k",
                "value": float(row["empirical_pass@k"]),
                "notes": notes,
            }
        )

    rows.append(
        {
            "method": oracle["method"],
            "metric": "oracle_accuracy",
            "value": oracle["oracle_accuracy"],
            "notes": oracle["reason"],
        }
    )

    return pd.DataFrame(rows)


def select_failure_examples(
    rows: Sequence[Dict[str, Any]],
    limit: int = 5,
) -> pd.DataFrame:
    """Return a small table of failed examples for quick inspection in the notebook."""
    _require_pandas()
    failed_rows = [row for row in rows if not row["passed"]]
    trimmed = []
    for row in failed_rows[:limit]:
        trimmed.append(
            {
                "task_id": row["task_id"],
                "status": row["status"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
                "raw_completion": textwrap.shorten(row["raw_completion"], width=180, placeholder=" ..."),
            }
        )
    return pd.DataFrame(trimmed)


def save_results_json(
    path: str,
    baseline: Dict[str, Any],
    baseline_rows: Sequence[Dict[str, Any]],
    pool_df: pd.DataFrame,
    passk_df: pd.DataFrame,
    oracle: Dict[str, Any],
) -> None:
    """Persist notebook outputs to JSON for later reporting."""
    _require_pandas()
    payload = {
        "baseline_metrics": baseline,
        "baseline_rows": list(baseline_rows),
        "candidate_pool": pool_df.to_dict(orient="records"),
        "pass_at_k": passk_df.to_dict(orient="records"),
        "oracle": oracle,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generate_candidates(
    model,
    tokenizer,
    prompt_text,
    num_samples=1,
    temperature=0.0,
    max_new_tokens=512,
    top_p=0.95,
):
    completions = []

    for _ in range(num_samples):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        completion_ids = outputs[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        completions.append(completion_text)

    return completions
