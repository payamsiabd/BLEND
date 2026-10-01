# 🚀 BLEND

This is the official repository for our paper
### Balancing Personalization vs. Generalization in Federated Vision–Language Models

🎉 **Accepted at NeurIPS 2026**


This repository is built on top of [PromptFL](https://github.com/PEILab-Federated-Learning/PromptFL). Please follow the steps below to properly set up the environment and required datasets.

---

## 🔍 Why Information Imbalance Matters

A key motivation behind BLEND is the **information imbalance between vision and text** in vision-language models.

Given latent scene semantics \(S\), visual observation \(V\), and a textual description \(T\) generated from the image, the information flow can be modeled as

\[
S \rightarrow V \rightarrow T,
\]

which implies, by the data-processing inequality,

\[
I(S;T) \leq I(S;V).
\]

This means that the visual modality can preserve richer semantic information than the corresponding textual description. In practice, an image may contain details about color, pose, background, and surrounding objects that are not reflected in a short text description. :chatgpt-content-reference{index="0"}

This imbalance is important for **personalized federated learning**. Since visual representations can contain a broader range of semantic factors, client-specific visual adaptation can act as a semantic filter that emphasizes locally relevant information. However, excessive personalization can hurt generalization because information that is unimportant for one client may still be useful for unseen classes or other clients. :chatgpt-content-reference{index="1"}

BLEND is designed around this trade-off: it preserves **globally shared visual and textual knowledge** while introducing a **local personalized visual branch**, allowing the model to adapt to each client without sacrificing its ability to generalize.


# ⚙️ Setup and Installation

## 🛠️ 1. Environment Setup

This project depends on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) framework.

First, install `dassl` and its dependencies by following the official installation instructions:

- 🔗 https://github.com/KaiyangZhou/Dassl.pytorch#installation

Make sure that the `dassl` environment is correctly installed and activated.

