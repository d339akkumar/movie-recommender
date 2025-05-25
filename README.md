# 🎬 MovieLens 1M Hybrid Recommender System

This project implements a **hybrid movie recommender system** using the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). It combines collaborative filtering, content-based filtering, and neural networks to address the **cold start problem** and provide personalized movie recommendations.

## 🚀 Features

- Collaborative Filtering:
  - Singular Value Decomposition (SVD)
- Content-Based Filtering:
  - TF-IDF vectorization of movie genres/descriptions
  - Cosine similarity-based recommendations
- Hybrid Recommender:
  - Combines content and collaborative scores
- Neural Network Autoencoder:
  - Learns latent features for better generalization
- Cold Start Handling:
  - Recommends based on genre/description similarity for new users or items
- Evaluation Metrics:
  - RMSE, MAE, Precision@10, Recall@10

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn, Surprise, LightFM
- Streamlit (for UI deployment)
- Matplotlib, Seaborn (for EDA)
## click on [link](https://deepak-movie-recommendation-ybejongj2zb5wdmz5macf4.streamlit.app/) to check out the deployed site
## Recommender.ipynb : Jupyter notebook with deep analysis and trained on various model
