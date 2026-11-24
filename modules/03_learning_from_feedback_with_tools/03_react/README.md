# ReAct — WebShop-Style Tasks

[Open the WebShop notebook](03_react_webshop.ipynb). It compares a greedy baseline, Act-only, and ReAct on nine synthetic products and six shopping goals. Follow the Thought → Action → Observation loop and evaluate exact task success, invalid actions, steps, and optional model usage. The real WebShop product preview is separate from the simulator evaluation.

For Jupyter or VS Code, start in this folder and install dependencies with `python -m pip install -r requirements.txt`. In Colab, the notebook's first cell clones or updates the repository and enters this directory. Change `REPO_BRANCH` in that cell if using a branch other than `main`.

After dependency setup, the simulator and greedy baseline run offline. To use a hosted model, set `RUN_LIVE_LLM = True` and provide `GROQ_API_KEY` or `OPENAI_API_KEY` through the environment or Colab Secrets.

[ReAct paper](https://arxiv.org/abs/2210.03629) · [WebShop](https://github.com/princeton-nlp/WebShop)
