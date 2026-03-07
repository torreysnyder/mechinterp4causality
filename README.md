# Exploring Causal Reasoning in Large Language Models via Causal Tracing

This repository contains the code accompanying the research project **"Exploring Causal Reasoning in Large Language Models via Causal Tracing"**, which investigates how causal structure is encoded and processed within transformer-based language models using mechanistic interpretability techniques.

---

## Overview

This project applies **causal tracing** and **ablation patching** to probe the internal mechanisms of large language models — specifically GPT-2-small — to understand how and where causal reasoning emerges within the model's activations. The pipeline covers fine-tuning, dataset generation, inference, and activation patching experiments.

---

## Repository Structure

```
.
├── fine_tune.py
├── Inference_via_next_token_prediction.py
├── activation_patching.py
└── causal_transformers/
    └── dataset/
        └── causal_dataset_gaussian.py
```

---

## Scripts

### `fine_tune.py`
Fine-tunes **GPT-2-small** on a causal reasoning dataset. This script handles training configuration, tokenization, and model checkpointing.

**Usage:**
```bash
python fine_tune.py
```

---

### `causal_transformers/dataset/causal_dataset_gaussian.py`
Generates **clean and corrupt prompt pairs** used in ablation patching experiments. Clean prompts contain valid causal structure, while corrupt prompts are perturbed via Gaussian noise to serve as counterfactual baselines.

**Usage:**
```bash
python causal_transformers/dataset/causal_dataset_gaussian.py
```

---

### `Inference_via_next_token_prediction.py`
Appends a **target token** to each prompt and runs inference via next-token prediction. This script prepares inputs for the patching pipeline by collecting logit distributions over candidate causal tokens.

**Usage:**
```bash
python Inference_via_next_token_prediction.py
```

---

### `activation_patching.py`
Runs **ablation patching experiments** by intervening on model activations at specific layers and positions. Patches activations from the clean run into the corrupt run to identify which components are causally responsible for the model's predictions.

**Usage:**
```bash
python activation_patching.py
```

---

## Pipeline

```
fine_tune.py
    ↓
causal_dataset_gaussian.py   →   clean / corrupt prompt pairs
    ↓
Inference_via_next_token_prediction.py   →   prompts with target token appended
    ↓
activation_patching.py   →   causal tracing results
```

---

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `transformers`
- `torch`
- `numpy`
- `datasets`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{causal_tracing_llms,
  title     = {Exploring Causal Reasoning in Large Language Models via Causal Tracing},
  year      = {2025},
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
