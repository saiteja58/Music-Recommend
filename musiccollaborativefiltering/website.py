import streamlit as st
from pathlib import Path
from data import load_user_artists, ArtistRetriever
from recommender import ImplicitRecommender
import implicit

# Title and description
st.title("Music Collaborative Filtering Recommendation")
st.write("This app recommends artists based on collaborative filtering.")

# Load artist data and model
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("C:\\Users\\saite\\OneDrive\\Desktop\\aiml\\musiccollaborativefiltering\\lastfmdata\\artists.dat"))
user_artists_matrix = load_user_artists(Path("C:\\Users\\saite\\OneDrive\\Desktop\\aiml\\musiccollaborativefiltering\\lastfmdata\\user_artists.dat"))

# Initialize ALS model
implicit_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)
recommender = ImplicitRecommender(artist_retriever, implicit_model)
recommender.fit(user_artists_matrix)

# User input for recommendations
user_id = st.number_input("Enter User ID", min_value=1, step=1)
n_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

# Get and display recommendations
if st.button("Get Recommendations"):
    try:
        artists, scores = recommender.recommend(user_id, user_artists_matrix, n=n_recommendations)
        st.write("Top Recommended Artists:")
        for artist, score in zip(artists, scores):
            st.write(f"{artist} - Score: {score}")
    except ValueError as e:
        st.error(f"Error: {e}")
