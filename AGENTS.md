# Agent Instructions for Self-Improving-Agent-Projects

## Project Goal

This repo is a portfolio-style independent study of Stanford CS329A Self-Improving Agents topics. New modules should match the style of modules 01 and 02: clear notebook narrative, reproducible experiments, helper functions, and concise portfolio-ready explanations.

## Required Structure for New Modules

When creating a new module, use this layout:

modules/03_topic_name/
  03_topic_name.ipynb
  helpers.py

The notebook should include:

1. Title and project goal
2. Dataset/task explanation
3. Baseline method
4. Improved/self-improving method
5. Evaluation metrics
6. Results table
7. Short interpretation
8. Portfolio summary

## Coding Style

- Keep notebooks readable for hiring managers.
- Move repeated logic into helpers.py.
- Use simple Python over heavy abstraction.
- Prefer clear parameter blocks near the top of notebooks.
- Include short markdown explanations before major code sections.
- Avoid unnecessary complexity.

## Experiment Standards

- Use frozen candidate pools when comparing methods.
- Avoid label leakage unless explicitly presenting an oracle upper bound.
- Clearly label deployable vs non-deployable methods.
- Track accuracy, pass rate, runtime, and token usage when relevant.
- Save important results as JSON or tables when practical.

## Do Not Commit

Do not commit:
- .venv/
- .codex/
- __pycache__/
- .ipynb_checkpoints/
- .env files
- model weights
- large generated datasets