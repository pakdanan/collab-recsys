import streamlit as st
import pandas as pd

# Load RecommenderSystem class (Assuming it's in the same file or imported from another module)
class RecommenderSystem:
    def __init__(self):
        self.df = None
        self.all_movies = []
        self.model = None

    def load_data(self, data_path):
        """Load the dataset from a pickle file."""
        self.df = pd.read_pickle(data_path)
        self.all_movies = self.df.movieId.unique()

    def load_model(self, model_path):
        """Load a trained model from a file."""
        self.model = pd.read_pickle(model_path)

    def recommend(self, user_id, topk=10):
        """Make recommendations for a given user."""
        if self.model is None or self.df is None:
            st.error("Model and data must be loaded first.")
            return pd.DataFrame()

        rated = self.df[self.df.userId == user_id].movie
        not_rated = [movieId for movieId in self.all_movies if movieId not in rated]
        score = [self.model.predict(user_id, movieId).est for movieId in not_rated]

        result = pd.DataFrame({"movieId": not_rated, "pred_score": score})
        result.sort_values("pred_score", ascending=False, inplace=True)
        return result.head(topk)

# Initialize RecommenderSystem
recommender = RecommenderSystem()

# Load the model and data pickle files
@st.cache_data
def load_data():
    recommender.load_data("recommender_data.pkl")

@st.cache_resource
def load_model():
    recommender.load_model("recommender_model.pkl")

@st.cache_resource
def load_title():
    movie_title = pd.read_pickle("movie_title.pkl")
    
# Streamlit UI
st.title("Movie Recommender System")
st.write("Enter a user ID to get personalized movie recommendations.")

# Load the model and data only once
load_data()
load_model()
load_title()

# User input for user ID and top-k recommendations
user_id = st.number_input("User ID", min_value=1, step=1)
topk = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

# Button to generate recommendations
if st.button("Get Recommendations"):
    recs = recommender.recommend(user_id=int(user_id), topk=topk)
    recs=recs.merge(movie_title,on='movieId')
    if not recs.empty:
        st.write("Top Recommendations:")
        st.dataframe(recs)
    else:
        st.write("No recommendations available for this user.")
