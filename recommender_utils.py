import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from difflib import get_close_matches
import re

def load_data():
    movies = pd.read_csv("data/movies.dat", sep="::", engine="python", header=None,
                         names=["movie_id", "title", "genres"], encoding='latin-1')
    ratings = pd.read_csv("data/ratings.dat", sep="::", engine="python", header=None,
                          names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin-1')
    users = pd.read_csv("data/users.dat", sep="::", engine="python", header=None,
                        names=["user_id", "gender", "age", "occupation", "zip"], encoding='latin-1')

    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    return movies, ratings, users

def preprocess_data(movies):
    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.replace('|', ' '))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movieid_to_idx = {mid: idx for idx, mid in enumerate(movies['movie_id'])}
    idx_to_movieid = {idx: mid for mid, idx in movieid_to_idx.items()}
    return cosine_sim, movieid_to_idx, idx_to_movieid

def load_svd_model():
    with open("models/svd_model.pkl", "rb") as f:
        return pickle.load(f)

def clean_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip().lower()

def recommend_by_title(movie_title, movies, cosine_sim, top_n=10):
    cleaned_input = movie_title.strip().lower()
    movies['clean_title'] = movies['title'].apply(clean_title)

    match = movies[movies['clean_title'] == cleaned_input]
    if match.empty:
        close_matches = get_close_matches(cleaned_input, movies['clean_title'].tolist(), n=1, cutoff=0.6)
        if not close_matches:
            return None
        match = movies[movies['clean_title'] == close_matches[0]]

    idx = match.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movie_id', 'title']]

def hybrid_recommend(user_id, movies, ratings, svd_model, cosine_sim, movieid_to_idx, top_n=10):
    if user_id not in ratings['user_id'].values:
        return None

    user_rated = ratings[ratings['user_id'] == user_id]
    top_rated = user_rated[user_rated['rating'] >= 4.0]
    user_top_indices = [movieid_to_idx[mid] for mid in top_rated['movie_id'] if mid in movieid_to_idx]

    predictions = []
    for movie_id in movies['movie_id'].unique():
        if movie_id in user_rated['movie_id'].values:
            continue
        try:
            svd_pred = svd_model.predict(user_id, movie_id).est
            movie_idx = movieid_to_idx.get(movie_id)
            if movie_idx is None:
                continue
            content_score = cosine_sim[movie_idx, user_top_indices].mean() if user_top_indices else 0
            hybrid_score = 0.7 * svd_pred + 0.3 * content_score
            predictions.append((movie_id, hybrid_score))
        except:
            continue

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in predictions[:top_n]]
    return movies[movies['movie_id'].isin(top_movie_ids)][['movie_id', 'title']]
