import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse  # Import sparse matrix checker

# Set data directory path (Cross-platform compatible)
DATA_DIR = os.path.join("D:\Guvi_Project\Swiggy’s Restaurant Recommendation System\Data")

# Function to load CSV files with caching for performance optimization
@st.cache_data
def load_csv(filename):
    """Loads a CSV file from the specified data directory."""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        st.error(f"{filename} file not found!")
        st.stop()
    return pd.read_csv(file_path)

# Function to load pickled files (models, encoders, scalers) with caching
@st.cache_resource
def load_pickle(filename):
    """Loads a pickled file (e.g., encoder, scaler) from the specified data directory."""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        st.error(f"{filename} file not found!")
        st.stop()
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load required datasets and pre-trained models
cleaned_df = load_csv("Cleaned_data.csv")  # Cleaned restaurant data
encoded_df = load_csv("Encoded_data.csv")  # Encoded features for similarity comparison
encoder = load_pickle("encoder.pkl")  # Pre-trained OneHotEncoder
scaler = load_pickle("scalar.pkl")  # Pre-fitted MinMaxScaler

# Streamlit UI Setup
st.title("🍽️ Swiggy’s Restaurant Recommendation System")
st.sidebar.header("🔍 User Preferences")

# Sidebar inputs for user preferences
selected_city = st.sidebar.selectbox("📍 Select City", cleaned_df["city"].unique())
selected_cuisine = st.sidebar.selectbox("🍛 Select Cuisine", cleaned_df["cuisine"].unique())
rating_preference = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
cost_preference = st.sidebar.slider("💰 Maximum Cost", int(cleaned_df["cost"].min()), int(cleaned_df["cost"].max()), int(cleaned_df["cost"].median()))
rating_count_preference = st.sidebar.slider("🗳️ Minimum Rating Count", 0, int(cleaned_df["rating_count"].max()), 10)

# Collect user inputs into a dictionary
user_input = {
    "city": selected_city,
    "cuisine": selected_cuisine,
    "rating": rating_preference,
    "cost": cost_preference,
    "rating_count": rating_count_preference
}

def find_nearest_restaurants(user_input, cleaned_df, encoder, scaler):
    """
    Finds the top 10 most similar restaurants based on user preferences using cosine similarity.
    
    Parameters:
        user_input (dict): Dictionary containing user preferences (city, cuisine, rating, cost, rating_count).
        cleaned_df (DataFrame): Original cleaned dataset containing restaurant information.
        encoder (OneHotEncoder): Pre-trained encoder for categorical features.
        scaler (StandardScaler): Pre-trained scaler for numerical features.

    Returns:
        DataFrame: Top 10 recommended restaurants.
    """
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Encode categorical features (city, cuisine) using OneHotEncoder
    transformed_cat_array = encoder.transform(user_df[['city', 'cuisine']])
    
    # Convert sparse matrix to dense array if necessary
    if issparse(transformed_cat_array):
        transformed_cat_array = transformed_cat_array.toarray()
    
    # Convert encoded array into a DataFrame with appropriate feature names
    transformed_cat_array_df = pd.DataFrame(transformed_cat_array, columns=encoder.get_feature_names_out())

    # Standardize numerical features (rating, rating_count, cost) using StandardScaler
    transformed_num_array = scaler.transform(user_df[['rating', 'rating_count', 'cost']])
    transformed_num_array_df = pd.DataFrame(transformed_num_array, columns=['rating', 'rating_count', 'cost'])

    # Combine encoded categorical features with standardized numerical features
    transformed_res = pd.concat([transformed_num_array_df, transformed_cat_array_df], axis=1)
    
    # Compute cosine similarity between user input and all restaurant feature vectors
    similarity_scores = cosine_similarity(encoded_df, transformed_res).flatten()

    # Get indices of the top 10 most similar restaurants
    nearest_idx = similarity_scores.argsort()[-10:][::-1]

    # Retrieve recommended restaurants based on similarity
    recommendations = cleaned_df.iloc[nearest_idx]
    
    # Reset index for better readability in Streamlit UI
    recommendations.reset_index(drop=True, inplace=True)
    recommendations.index += 1  # Start index from 1 instead of 0
    
    return recommendations

# Display recommendations when the user clicks the button
if st.sidebar.button("🎯 Get Recommendations"):
    recommendations = find_nearest_restaurants(user_input, cleaned_df, encoder, scaler)
    st.success("✅ Here are the best matches for you!")
    st.dataframe(recommendations, use_container_width=True)
