import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle

ratings = pd.read_csv("data/ratings.dat", sep="::", engine="python", header=None,
                      names=["user_id", "movie_id", "rating", "timestamp"], encoding='latin-1')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)
