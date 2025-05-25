import streamlit as st
from recommender_utils import *

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_all():
    movies, ratings, users = load_data()
    cosine_sim, movieid_to_idx, idx_to_movieid = preprocess_data(movies)
    svd_model = load_svd_model()
    return movies, ratings, users, cosine_sim, movieid_to_idx, svd_model

movies, ratings, users, cosine_sim, movieid_to_idx, svd_model = load_all()

option = st.radio("Choose recommendation type:", ["By User ID", "By Movie Title"])

if option == "By User ID":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    if st.button("Get Recommendations"):
        user_id = int(user_id)  # Ensure correct type
        recs = hybrid_recommend(user_id, movies, ratings, svd_model, cosine_sim, movieid_to_idx)
        if recs is not None and not recs.empty:
            st.write("🎯 Recommended Movies:")
            st.dataframe(recs)
        else:
            st.warning("No recommendations found.")

elif option == "By Movie Title":
    title_input = st.text_input("Enter Movie Title")
    if st.button("Find Similar Movies"):
        recs = recommend_by_title(title_input, movies, cosine_sim)
        if recs is not None and not recs.empty:
            st.write(f"🎬 Movies similar to '{title_input}':")
            st.dataframe(recs)
        else:
            st.warning("Movie not found or no recommendations.")
