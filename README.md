# Self-Improving Agents

This project is a compact, hands-on exploration of **test-time compute (TTC)** for reasoning models.

The central idea is simple and timely: if a model is small, can you still push its performance upward by changing how you use it at inference time rather than by training a bigger model?

Instead of treating inference as a single forward pass, this project treats it as a design space. It compares one-shot generation, repeated sampling, revision loops, and lightweight search on GSM8K.

## Why This Is Worth Reading

What makes this project interesting is that it is not just a collection of utilities or notebook fragments. It is one structured experiment with a clear narrative arc:

1. establish a baseline
2. spend more compute in simple ways
3. measure where that extra compute helps
4. test more structured strategies
5. compare the resulting TTC tradeoffs clearly

That progression makes the project easy to follow even if the reader is new to TTC.

## What This Repo Contains

The active project is:

- `modules/01_test_time_compute/01-test-time-compute.ipynb`
- `modules/01_test_time_compute/helpers.py`

The notebook is the main story. The helper module contains the reusable generation, selection, evaluation, revision, and search utilities so the notebook can stay focused on explanation rather than plumbing.

## What The Notebook Covers

The notebook is structured as one experiment with five steps:

1. Baseline single-pass generation
2. Best-of-N with majority vote
3. Difficulty estimation from sampled pass rate
4. Sequential revision chains
5. Lightweight beam-style search

The notebook is built to answer one practical question:

How much can a small model improve if you spend extra compute at inference time, and which TTC strategy gives the best tradeoff between accuracy and cost?

## Recommended Reading Order

For the clearest experience:

1. Read this `README`
2. Open `modules/01_test_time_compute/01-test-time-compute.ipynb`
3. Use `modules/01_test_time_compute/helpers.py` only when you want implementation details

If you are new to the project, do not start in `helpers.py`. The notebook is intentionally written to explain the ideas first and the implementation second.

## Quick Start

The easiest way to run the project is in Google Colab with a GPU runtime, but the repo now also includes a lightweight dependency file for local reproduction.

### Colab

Clone the repo:

```bash
!git clone https://github.com/dariuscparker/Self-Improving-Agent-Projects.git
%cd Self-Improving-Agent-Projects
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
modules/01_test_time_compute/01-test-time-compute.ipynb
```

The notebook still includes an in-notebook install cell for Colab convenience and writes artifacts to:

```text
.codex/artifacts/
```

### Local

Local execution is also possible if you have a Python 3 environment with the packages listed in `requirements.txt`.

You will still want a GPU for a practical run.

## Reproducibility

For the clearest reproduction path:

1. Create a fresh environment.
2. Run `pip install -r requirements.txt`.
3. Open `modules/01_test_time_compute/01-test-time-compute.ipynb`.
4. Run the notebook top to bottom.

Methods that use sampling (`TEMPERATURE > 0`) are stochastic. For stronger reporting, rerun those sections with multiple seeds and report the spread instead of a single lucky run.

## Project Structure

```text
Self-Improving-Agents/
├── README.md
└── modules/
    └── 01_test_time_compute/
        ├── 01-test-time-compute.ipynb
        └── helpers.py
```

## What To Expect From A Run

The notebook:

- loads GSM8K
- loads a quantized `Qwen/Qwen3-1.7B` model
- evaluates several TTC strategies on a shared subset
- compares those strategies using accuracy, total-token cost, and runtime
- saves summary artifacts for later analysis

Some sections are intentionally more expensive than others. If you want a faster first pass, reduce `MAX_QUESTIONS`, `N_SAMPLES`, `REVISION_STEPS`, or beam settings near the top of the notebook.

## Who This Is For

This project should be useful to readers who want to:

- understand TTC without starting from a large research codebase
- see multiple inference-time strategies in one place
- compare those strategies on a concrete benchmark
- study a notebook that is organized around a single question rather than a grab bag of experiments

## Why The Repo Is Structured This Way

The project is deliberately split into:

- notebook for explanation and experiment flow
- helper file for reusable logic

That separation keeps the narrative readable while still making the implementation easy to inspect and reuse.
