# 📊 Cutting-Edge Research in Graph Neural Networks (GNNs)

## Overview

This project explores **real-world applications of Graph Neural Networks (GNNs)** using a large-scale social graph dataset from **LastFM**. The core objective is to design, implement, and evaluate advanced GNN architectures for **friend recommendation via link (edge) prediction**, based on shared music preferences.

The work was completed as part of **COMP8221 – Advanced Machine Learning** and focuses on both **model innovation** and **rigorous experimental analysis**.

---

## Problem Statement

Music streaming platforms such as LastFM and Spotify rely heavily on understanding **user–item interaction graphs** to drive recommendations and social engagement.

In this project:

* Users and artists are modeled as nodes
* Interactions (listening history) are modeled as edges
* The task is framed as an **edge prediction problem**:

  > Predict potential new user–user connections based on shared music interests

---

## Dataset

* **Dataset:** LastFM user–artist interaction graph
* **Graph Type:** Bipartite graph (Users ↔ Artists)
* **Scale Handling:** Subgraph sampling to balance performance and computation

### Graph Representation

* Nodes: Users, Artists
* Edges: User listening interactions
* Features:

  * Synthetic 64-dimensional node embeddings for nodes without attributes

---

## Model Architecture

The primary contribution of this project is a **custom hybrid GNN model**:

### 🔹 AdvancedFriendRecGNN

A multi-layer architecture combining the strengths of different GNN paradigms:

1. **GCN Layer** – Captures basic neighborhood structure
2. **GraphSAGE Layer** – Improves scalability via neighbor sampling
3. **GAT Layer (Multi-Head Attention)** – Learns weighted neighbor importance
4. **Custom Neighbor Importance Layer** – Fine-grained attention mechanism
5. **Final GCN Layer** – Produces cohesive node embeddings

Additional techniques:

* Batch Normalization
* Edge Dropout
* Contrastive Loss
* Gradient Clipping
* Cosine Annealing Learning Rate Scheduler

---

## Training Strategy

* **Edge Split:**

  * 60% Training
  * 20% Validation
  * 20% Testing
* **Negative Sampling:** Used to distinguish real vs non-existent edges
* **Regularization:** Dropout and edge dropout to reduce overfitting

---

## Evaluation Metrics

The models were evaluated using standard link-prediction metrics:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.586 |
| Precision | 0.555 |
| Recall    | 0.867 |
| F1 Score  | 0.677 |

📌 High recall demonstrates the model’s effectiveness in identifying relevant connections, which is critical in recommendation systems.

---

## Ablation Study

To understand model behavior, multiple ablations were performed:

* **No Attention**
* **No Batch Normalization**
* **No Contrastive Loss**

### Key Insight

> Simpler architectures (e.g., GCN-only) sometimes outperform complex models, highlighting the importance of **model interpretability and component selection** in graph learning tasks.

---

## Model Comparisons

The project compares:

* AdvancedFriendRecGNN (Custom)
* Basic GCN
* GraphSAGE
* GAT
* Multiple ablated variants

This provides a **comprehensive benchmark** across architectures and learning strategies.

---

## Key Learnings

* Attention mechanisms do not always guarantee performance gains
* Batch normalization is critical for training stability
* Graph structure complexity significantly impacts precision–recall trade-offs
* GNN design requires careful balance between **model complexity and generalization**

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* NumPy, Pandas
* Matplotlib / Seaborn
* Jupyter Notebook

---

## Future Improvements

* Incorporate real node features instead of synthetic embeddings
* Explore temporal graph neural networks (TGNNs)
* Deploy model as a recommendation microservice
* Extend to heterogeneous graphs with richer relations

---

## Author

**Gopi Abhiram Vishal Chongala**
Master of Information Technology (Artificial Intelligence)
Macquarie University
