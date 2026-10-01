# 🚀 BLEND

This is the official repository for our paper
### Balancing Personalization vs. Generalization in Federated Vision–Language Models

🎉 **Accepted at NeurIPS 2026**


This repository is built on top of [PromptFL](https://github.com/PEILab-Federated-Learning/PromptFL). Please follow the steps below to properly set up the environment and required datasets.

---

## 🔍 Motivation

A key motivation behind BLEND is the **information imbalance between vision and text** in vision-language models. Visual representations often preserve a broader range of semantic information, while text captures only a subset of those details. This makes the visual branch a natural place for stronger client-specific adaptation, while preserving globally transferable knowledge remains important for generalization. 

## ✨ Key Idea

BLEND is built around the idea that **personalization and generalization should not be treated symmetrically across modalities**.

- The **visual modality** contains richer and more diverse semantic information, making it especially suitable for client-specific adaptation.
- At the same time, globally shared knowledge must be preserved so that personalization does not come at the cost of performance on unseen classes.
- BLEND therefore introduces **asymmetric personalization**, where client-specific adaptation is emphasized on the visual side while globally shared visual and textual representations are maintained.
- By combining personalized and global visual representations, BLEND aims to retain information that is important for each client while still preserving transferable knowledge across the federation.
- ⚖️ This design directly targets the central challenge in personalized federated vision-language learning: **improving local adaptation without sacrificing generalization**.

---

# ⚙️ Setup and Installation

## 🛠️ 1. Environment Setup

This project depends on the [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) framework.

First, install `dassl` and its dependencies by following the official installation instructions:

- 🔗 https://github.com/KaiyangZhou/Dassl.pytorch#installation

Make sure that the `dassl` environment is correctly installed and activated.

