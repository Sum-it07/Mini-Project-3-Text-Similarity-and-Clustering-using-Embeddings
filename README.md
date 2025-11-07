# Mini Project 3 — Text Similarity and Clustering using Embeddings

## Overview
This project explores **text similarity** and **unsupervised clustering** using sentence embeddings. It leverages modern NLP embeddings from Hugging Face to group semantically similar sentences and evaluate clustering quality. The aim is to understand how textual data can be represented numerically and how semantic similarity can be captured using vector-based methods.

---

##  Objectives
- Compute similarity scores between pairs of text samples.
- Identify top-k similar sentences for a given query.
- Cluster textual data into categories using embedding vectors.
- Evaluate clustering accuracy against known labels.

---

##  Technologies Used
- **Python 3**
- **Hugging Face Transformers / Sentence Transformers**
- **scikit-learn** (for K-Means, metrics)
- **NumPy / Pandas**
- **Matplotlib / Seaborn** (for visualizations)
- **Cosine Similarity** (custom + prebuilt methods)

---

## Methodology
1. **Data Preprocessing**
   - Load text dataset (e.g., news articles or reviews).
   - Clean and tokenize text.
2. **Text Embedding**
   - Generate embedding vectors using a pre-trained transformer model.
3. **Similarity Computation**
   - Define a custom function to calculate **cosine similarity** between two text embeddings.
   - Retrieve **top-k most similar sentences** for a given query.
4. **Clustering**
   - Apply **K-Means clustering** to group texts based on semantic similarity.
   - Assign each document a cluster label.
5. **Evaluation**
   - Compare predicted clusters with true categories.
   - Generate a **classification report** to measure accuracy and misclassifications.

---

##  Results
- Successfully identified semantically coherent clusters.
- High similarity observed among related text samples.
- Quantitative metrics (precision, recall, F1-score) show strong correlation between embeddings and true labels.

---

##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Sum-it07/Mini-Project-3-Text-Similarity-and-Clustering-using-Embeddings.git
   cd Mini-Project-3-Text-Similarity-and-Clustering-using-Embeddings

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Run the notebook:
  ```bash
jupyter notebook Mini_project3.ipynb
  ```

## Key Functions

cosine_score(text1, text2) → Computes similarity between two texts.

top_k_similar_sentences(embedding_matrix, query_text, k) → Finds the top k similar sentences.

K-Means model → Clusters the embeddings into meaningful groups.

## Author

[Sumit Shrestha]
Machine Learning Research Enthusiast
📧 sumitstha2060@gmail.com

