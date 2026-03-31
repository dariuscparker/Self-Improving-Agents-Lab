# Self-Improving Agent Projects

A collection of projects and experiments based on Stanford’s *Self-Improving Agents* coursework (CS239A), focused on modern techniques in agentic AI, test-time compute, and verification.

---

## 📌 Overview

This repository documents my independent study and implementation of concepts from Stanford’s Self-Improving Agents curriculum. The goal is to:

- Stay current with emerging techniques in agentic AI
- Implement and evaluate test-time compute (TTC) strategies
- Explore verification, refinement, and search-based reasoning methods
- Build practical, reproducible experiments using modern LLM tooling

This work emphasizes both **research understanding** and **hands-on implementation**.

---

## 🎓 Course Reference

- Official course page: https://cs329a.stanford.edu/
- Lecture videos: https://www.youtube.com/playlist?list=PL3058ht9NqT1NG6Y663elpHSDh-AW1TIr

> Note: Exercises and implementations in this repository are self-directed and inspired by the course topics. Exercises were generated with assistance from ChatGPT to align with the curriculum. Users are highly encouraged to complete the suggested readings In the curriculum to best understand the topics covered.  

---

## 🧠 Topics Covered

This repository follows the core themes of Stanford’s *Self-Improving Agents* course and explores the following areas:

### 🔁 Self-Improvement & Scaling
- Test-time compute scaling (Best-of-N, sampling strategies)
- Verifier-based selection (process vs outcome supervision)
- Constitutional AI and learning from AI feedback
- Train-time scaling and reinforcement learning for reasoning

### ✅ Verification & Evaluation
- Outcome and process verifiers
- Weak supervision and automated verification
- Evaluation frameworks for agent performance
- Long-horizon and real-world task evaluation

### 🧩 Reasoning, Planning & Search
- Multi-step reasoning and decomposition
- Sequential refinement / self-improvement loops
- Search-based reasoning (beam search, tree-based methods)
- Combining search with LLM inference

### 🛠️ Tool Use & Agent Capabilities
- Tool-augmented LLMs (ReAct-style agents)
- Code execution and feedback loops
- Memory-augmented agents
- Retrieval and external knowledge integration

### 🤖 Agent Systems & Architectures
- Agent orchestration and workflows
- Autonomous research and coding agents
- Open-ended self-improving systems
- Agent design and system-level optimization

### 🧪 Advanced Topics & Applications
- AI agents for software engineering
- Scientific discovery agents
- Multimodal and robotics-based agents
- Real-world agent deployment challenges

---

## 🚀 Getting Started

The recommended way to use this project is in **Google Colab** so you can run notebooks and experiments without needing to configure a local Python environment first.

### 1. Open Google Colab

- Go to https://colab.research.google.com/
- Start a new notebook
- Use a Python 3 runtime

### 2. Clone This Repository in Colab

Run the following command in a Colab cell:

```bash
!git clone https://github.com/dariuscparker/Self-Improving-Agent-Projects.git
```

Then move into the project directory:

```bash
%cd Self-Improving-Agent-Projects
```

### 3. Work From the Repository

After cloning, you can:

- Open and run notebooks directly in Colab
- Install any project-specific dependencies in notebook cells
- Save your outputs back to GitHub or Google Drive as needed

If a notebook or experiment requires additional packages, install them from a Colab cell with:

```bash
!pip install -r requirements.txt
```

If a `requirements.txt` file is not present for a specific project, install only the packages needed for that notebook or experiment.

---