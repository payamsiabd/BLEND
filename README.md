# 🚀 BLEND

This is the official repository for our paper
### Balancing Personalization vs. Generalization in Federated Vision–Language Models

🎉 **Accepted at NeurIPS 2026**


This repository is built on top of [PromptFL](https://github.com/PEILab-Federated-Learning/PromptFL). Please follow the steps below to properly set up the environment and required datasets.

---

## 🔍 Motivation and Method

A key motivation behind BLEND is the **information imbalance between vision and text** in vision-language models. Visual representations often preserve a broader range of semantic information, while text captures only a subset of those details. This makes the visual branch a natural place for stronger client-specific adaptation, while preserving globally transferable knowledge remains important for generalization. :chatgpt-content-reference{index="0"}

## 🧠 How BLEND Works

BLEND addresses this through an **asymmetric personalization strategy**:

- 🌐 **Global vision and text adapters** are shared and aggregated across clients to capture transferable knowledge.
- 🎯 A **personalized vision projection adapter** remains local to each client to learn client-specific representations.
- 🔀 The global and personalized visual representations are **fused** to balance personalization and generalization.
- ⚓ An **anchor loss** helps preserve generalizable representations during local training. :chatgpt-content-reference{index="1"} :chatgpt-content-reference{index="2"}

## ✨ Key Idea

BLEND leverages richer visual information for **client-specific personalization** while preserving shared visual and textual knowledge for **generalization to unseen classes**.

---

# ⚙️ Setup and Installation

## 🛠️ 1. Environment Setup

This project depends on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) framework.

First, install `dassl` and its dependencies by following the official installation instructions:

- 🔗 https://github.com/KaiyangZhou/Dassl.pytorch#installation

Make sure that the `dassl` environment is correctly installed and activated.

