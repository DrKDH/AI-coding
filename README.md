# ECIA: Emotional-Cognition Integration Architecture

A biologically-inspired reinforcement learning framework that demonstrates emotions are not obstacles but sophisticated computational mechanisms for adaptive decision-making in uncertain environments.

## 🧠 Key Features

- **8-Dimensional Emotional System**: Joy, Fear, Hope, Sadness, Curiosity, Anger, Pride, and Shame
- **Hippocampal Memory Module**: Context-aware episodic memory with similarity-based retrieval
- **Dopamine-Based Learning**: Adaptive learning rates based on prediction errors
- **Multi-Core Decision Making**: Parallel decision processes with dynamic weighting

## 📊 Performance

ECIA significantly outperforms traditional RL algorithms (ε-Greedy, UCB, Thompson Sampling) in non-stationary environments:
- **Environment A**: 0.82±0.05 mean reward (vs UCB's 0.81±0.07, p < 0.001)
- **Environment C**: 0.69±0.07 mean reward (vs UCB's 0.65±0.10, p < 0.001)

## 🚀 Quick Start

```python
# Run all experiments
from ECIA_coding import run_all_environments, AGENTS, ENVIRONMENTS
run_all_environments(AGENTS, ENVIRONMENTS)
