# Self-Improving Agent Projects

Independent, notebook-based implementations inspired by Stanford CS329A:
Self-Improving AI Agents.

This repository is a work in progress. The first two modules are present now:
test-time compute scaling and robust verification. Remaining sections are
planned from the CS329A syllabus at https://cs329a.stanford.edu/.

## Course Reference

- Official course page: https://cs329a.stanford.edu/
- Course: Stanford CS329A, Self-Improving AI Agents
- Term referenced by the public syllabus: Autumn 2025

The course covers self-improving AI agents through topics such as scaling
test-time compute, verifier-based selection, learning from feedback, tool use,
planning, train-time scaling with reinforcement learning, memory, agentic
software engineering, robust evaluation, multimodal agents, robotics, and
future research directions.

## Repository Status

| Module | Status | Local files | Focus |
| --- | --- | --- | --- |
| 01 - Test-Time Compute | In progress / implemented | `modules/01_test_time_compute/` | GSM8K reasoning with baseline sampling, self-consistency, difficulty bins, and sequential revision chains |
| 02 - Robust Verification | In progress / implemented | `modules/02_robust_verification/` | HumanEval code generation with pass@k, oracle best-of-N, public-test verifiers, and LLM judge selection |
| 03 - Learning from Feedback with Tools/Code | Planned | TBD | Execution feedback, tool use, ReAct-style loops, and constitutional or AI-feedback methods |
| 04 - Multi-Step Reasoning and Planning | Planned | TBD | Decomposition, planning, tree search, adaptive branching, and agentic workflows |
| 05 - Train-Time Scaling / RL | Planned | TBD | Self-training, reinforcement learning for reasoning, and scalable post-training systems |
| 06 - Open-Ended Evolution of Agents | Planned | TBD | Automated agent design, AI scientist-style loops, and open-ended improvement |
| 07 - Search and Deep Research Agents | Planned | TBD | Search-augmented reasoning, research agents, and code-generation search systems |
| 08 - Agentic Software Engineering | Planned | TBD | Coding agents, software benchmarks, kernel optimization, and agent-system interfaces |
| 09 - Memory-Augmented Agents | Planned | TBD | Long-context representations, memory systems, and cached knowledge mechanisms |
| 10 - Agentic Evaluations and Long-Horizon Tasks | Planned | TBD | Long-task benchmarks, economic task evaluation, research synthesis evaluation, and reliability analysis |
| 11 - Multimodal and Robotics Agents | Planned | TBD | Multimodal interaction, autonomous systems, and robotics-oriented agents |
| 12 - Future Research Areas | Planned | TBD | Open problems and synthesis across the course themes |

## Current Modules

### 01 - Test-Time Compute

The first notebook studies how additional inference-time work can improve a
small model's reasoning on GSM8K. It uses Hugging Face models and datasets to
compare baseline generation against multi-sample selection and revision-based
self-improvement.

Key ideas:

- Baseline single-sample generation
- Frozen proposal pools
- Self-consistency / majority-vote selection
- Difficulty estimates from pass rate and answer agreement
- Targeted sequential revisions for harder questions
- Accuracy and token-use summaries

### 02 - Robust Verification

The second notebook studies verifier-based selection for code-generation tasks
on HumanEval. It separates generation quality from selection quality by
evaluating several selectors on the same frozen candidate pool.

Key ideas:

- Baseline pass@1
- Candidate pools and pass@k
- Oracle best-of-N as an upper bound
- Public-test verifier selection
- LLM-as-judge selection
- Failure analysis and result artifacts

## Setup

These notebooks are designed primarily for Google Colab or another GPU-backed
notebook environment. Local execution is possible, but model loading may require
substantial memory depending on the selected model.

### Google Colab

```bash
!git clone https://github.com/dariuscparker/Self-Improving-Agent-Projects.git
%cd Self-Improving-Agent-Projects
!pip install -r requirements.txt
```

Then open a notebook from one of the module directories and run it top to
bottom.

### Local Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch your preferred notebook environment and open the module notebook.

## Requirements Notes

The shared `requirements.txt` reflects the dependencies used by the current
notebooks and helper modules. Colab may already include some packages, but
installing from `requirements.txt` keeps the runtime closer to the project
environment.

For quantized model loading, `bitsandbytes` works best in Linux/Colab GPU
runtimes. On CPU-only machines or unsupported platforms, you may need to adjust
the notebook model-loading cells before running large models.

## Project Notes

This is a self-directed study project inspired by CS329A rather than an
official Stanford course repository. The notebooks are intended to be read as
experiments: they include setup, implementation notes, evaluation tables, and
result summaries alongside the code.

