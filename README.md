# Refining GNN Models Using Attention-Based Aggregation Methods

An implementation and comparative analysis of embedding aggregation strategies within GNN model architectures to reconstruct Gene Regulatory Networks (GRNs) from multi-instance spatial gene expression data. 

## 📄 Project Whitepaper
The comprehensive project report detailing the literature review, system architecture, methodology, and performance charts is available directly in this repository:
👉 **[View Full Technical Report PDF](./GNN%20Project.pdf)**

---

## 🔬 Architectural Overview
The system processes multi-instance visual genomics data using a decoupled two-stage pipeline:
1. **Stage I (Feature Extraction):** A Siamese Convolutional Neural Network trained via contrastive learning maps high-dimensional, multi-view (dorsal, lateral, ventral) spatial gene expression images into compact latent embeddings.
2. **Stage II (Aggregation & Topology Inference):** Unordered instance embeddings are passed through deep structural aggregators (evaluating a learnable LSTM sequence layer vs. static Mean/Max pooling) to form unified gene node features. These features are then evaluated across various Graph Neural Network encoders and link prediction heads to predict regulatory network edges.

---

## 🗂️ Codebase Architecture

This repository is built using a clean, modular pipeline:
* `train.py`: The primary training pipeline loop managing multi-epoch data iteration, parameter optimization, and performance tracking.
* `test.py`: Validation evaluation script that loads saved weights checkpoints, evaluates models across test sets, and exports performance bar charts.
* `model.py`: Core architecture definitions for structural GNN layers (**GraphSAGE, GAT, GCN**) paired with custom edge link prediction heads (**MLP, Dot Product, Linear Predictors**).
* `graph.py`: Data loaders handling random permutations, graph layout splitting (train/validation/test), and structuring topological maps inside the Deep Graph Library (DGL).
* `aggregation.py`: Custom multi-instance pooling layers (**LSTM, Mean, Max**) designed to aggregate multi-view spatial representations while forcing permutation-invariance.
* `utils.py`: Analytical helpers computing evaluation metrics (Accuracy, F1-Score, ROC-AUC) and setting up linear warmup cosine learning rate schedulers.

---

## 📊 Core Performance Metrics
Evaluated using benchmark datasets (including eye and mesoderm development networks), our results demonstrate that learnable sequence aggregation models outpace standard static pooling variants:

| Aggregator Variant | Best Accuracy | Best F1-Score | Best ROC-AUC |
| :--- | :---: | :---: | :---: |
| **LSTM Aggregator** | **0.811** | **0.811** | **0.865** |
| Max Pooling | 0.768 | 0.754 | 0.812 |
| Mean Pooling | 0.782 | 0.771 | 0.829 |

---

## 🛠️ Quickstart Installation

Ensure you have your environment requirements ready:
```bash
pip install torch dgl scikit-learn matplotlib numpy
