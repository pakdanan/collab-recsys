import streamlit as st
import pandas as pd
import pickle

# Load RecommenderSystem class (Assuming it's in the same file or imported from another module)
class RecommenderSystem:
    def __init__(self):
        self.df = None
        self.all_movies = []
        self.model = None

    def load_data(self, data_path):
        """Load the dataset from a pickle file."""
        with open(data_path, 'rb') as f:
            self.df = pickle.load(f)
            self.all_movies = self.df.movieId.unique()

    def load_model(self, model_path):
        """Load a trained model from a file."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def recommend(self, user_id, topk=10):
        """Make recommendations for a given user."""
        if self.model is None or self.df is None:
            st.error("Model and data must be loaded first.")
            return pd.DataFrame()

        rated = self.df[self.df.userId == user_id].movieId
        not_rated = [movieId for movieId in self.all_movies if movieId not in rated]
        score = [self.model.predict(user_id, movieId).est for movieId in not_rated]
        result = pd.DataFrame({"movieId": not_rated, "pred_score": score})
        result.sort_values("pred_score", ascending=False, inplace=True)
        # merge with movie_title.pkl to get movie title:
        with open("movie_title.pkl", 'rb') as f:
            movie_title = pickle.load(f)
        result=result.merge(movie_title,on='movieId')
        return result.head(topk)

# Streamlit UI
st.title("Movie Recommender System")
st.write("Enter a user ID to get personalized movie recommendations.")

# User input for user ID and top-k recommendations
user_id = st.number_input("User ID", min_value=1, step=1)
topk = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

# Initialize RecommenderSystem
recommender = RecommenderSystem()

# Button to generate recommendations
if st.button("Get Recommendations"):
    recommender.load_data("recommender_data.pkl")
    recommender.load_model("recommender_model.pkl")
    recs = recommender.recommend(user_id=int(user_id), topk=topk)
    if not recs.empty:
        st.write("Top Recommendations:")
        st.dataframe(recs)
    else:
        st.write("No recommendations available for this user.")
