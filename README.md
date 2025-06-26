# 🧠 GPT2-Small Spam Classifier — *Built from Scratch & Fine-Tuned*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Model](https://img.shields.io/badge/Model-GPT2_Small_124M-purple?logo=openai)
![Tokenizer](https://img.shields.io/badge/Tokenizer-tiktoken-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)

---

## 📝 Table of Contents

- [📌 About the Project](#-about-the-project)
- [📉 Model Performance](#-model-performance)
- [🧠 Architecture Overview](#-architecture-overview)
  - [🧱 Before Fine-Tuning](#-before-fine-tuning)
  - [🛠️ After Fine-Tuning](#-after-fine-tuning)
- [🛠️ Requirements](#️-requirements)
- [🚀 How to Use](#-how-to-use)
- [🧪 Sample Inference](#-sample-inference)
- [📘 Credits](#-credits)
- [📄 License](#-license)
- [📧 Contact](#-contact)

---

## 📌 About the Project

This project demonstrates how to **build a GPT2-Small (124M parameters) transformer model from scratch using PyTorch**, tokenize input using OpenAI's `tiktoken`, and fine-tune the model for **binary text classification** (spam vs. not spam).

Key Highlights:

- Implements Transformer, MultiHeadAttention, LayerNorm, and PositionalEncoding from scratch.
- Integrates a classification head for binary prediction.
- Achieves high accuracy and generalization across datasets.
- Includes visualizations for accuracy/loss and architectural diagrams.

---

## 📉 Model Performance

| Dataset    | Accuracy (%) | Loss     |
|------------|--------------|----------|
| **Train**      | 98.75%       | 0.772    |
| **Validation** | 95.97%       | 0.764    |
| **Test**       | 94.00%       | 0.789    |

> 📈 The model generalizes well across unseen data and maintains high precision.

---

## 🧠 Architecture Overview

### 🧱 Before Fine-Tuning
```mermaid
flowchart TD

%% ==== Styling (must come first for GitHub Mermaid) ====
classDef gpt fill=#e6f2ff,stroke=#3399ff,stroke-width:1px;
classDef clf fill=#fff4e6,stroke=#ff9933,stroke-width:1px;
class D1,E1,F1,G1 gpt
class D2,E2,F2,G2 clf

%% ==== BEFORE FINE-TUNING ====
subgraph "Original GPT-2 (Before Fine-Tuning)"
    A1["Input Text"] --> B1["Tokenizer (tiktoken)"]
    B1 --> C1["Token IDs"]
    C1 --> D1["Embedding Layer (Token + Positional)"]
    D1 --> E1["Transformer Blocks ×12"]
    E1 --> F1["Final LayerNorm"]
    F1 --> G1["Vocabulary Logits (50257 classes)"]
    G1 --> H1["Next Token Prediction"]
end

%% ==== AFTER FINE-TUNING ====
subgraph "Fine-Tuned GPT for Classification (After Fine-Tuning)"
    A2["Input Text"] --> B2["Tokenizer (tiktoken)"]
    B2 --> C2["Token IDs"]
    C2 --> D2["Embedding Layer (Token + Positional)"]
    D2 --> E2["Transformer Blocks ×12"]
    E2 --> F2["Final LayerNorm"]
    F2 --> G2["Classification Head (Linear 768 → 2)"]
    G2 --> H2["Output Label (spam / not spam)"]
end
```
