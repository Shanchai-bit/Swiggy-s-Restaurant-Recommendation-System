from jwt import encode
import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Cache the function to load the cleaned data to avoid reloading on each interaction
@st.cache_resource
def load_cleaned_data():
    cleaned_df = pd.read_csv(r"D:\Guvi_Project\Swiggy’s Restaurant Recommendation System\Data\Cleaned_data.csv")
    return cleaned_df

# Cache the function to load the encoded data to avoid reloading on each interaction
@st.cache_resource
def load_encoded_data():
    encoded_df = pd.read_csv(r"D:\Guvi_Project\Swiggy’s Restaurant Recommendation System\Data\Encoded_data.csv")
    return encoded_df

# Cache the trained KMeans model so it isn't reloaded multiple times
@st.cache_resource
def load_kmeans_model():
    kmeans_file_path = r"D:\Guvi_Project\Swiggy’s Restaurant Recommendation System\Data\trained_model_file.pkl"
    with open(kmeans_file_path, "rb") as f:
        kmeans = pickle.load(f)
    return kmeans

# Cache the OneHotEncoder so it's not reloaded multiple times
@st.cache_resource
def load_encoder():
    encoder_file_path = r"D:\Guvi_Project\Swiggy’s Restaurant Recommendation System\Data\encoder.pkl"
    with open(encoder_file_path, "rb") as f:
        encoder = pickle.load(f)
    return encoder

# Streamlit UI
st.title("Restaurant Recommendation System")
st.sidebar.header("User Preferences")

# User Inputs from session state (if exists)
if 'user_input' not in st.session_state:
    st.session_state.user_input = {
        "city": None,
        "cuisine": None,
        "rating": 3.0,
        "cost": 0,
        "rating_count": 0
    }

# Load data and models (only once)
cleaned_df = load_cleaned_data()
encoded_df = load_encoded_data()
kmeans = load_kmeans_model()
encoder = load_encoder()

# Assign clusters to cleaned data (done once)
if 'cluster' not in st.session_state:
    cleaned_df["cluster"] = kmeans.labels_
    st.session_state.cluster = cleaned_df["cluster"]

# Sidebar for user input
selected_city = st.sidebar.selectbox("Select City", cleaned_df["city"].unique(), index=0)
selected_cuisine = st.sidebar.selectbox("Select Cuisine", cleaned_df["cuisine"].unique(), index=0)
rating_preference = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
cost_preference = st.sidebar.slider("Maximum Cost", int(cleaned_df["cost"].min()), int(cleaned_df["cost"].max()), int(cleaned_df["cost"].median()))
rating_count_preference = st.sidebar.selectbox("Minimum Rating Count", cleaned_df["rating_count"].unique())

# Update session state with user inputs
if selected_city != st.session_state.user_input['city']:
    st.session_state.user_input["city"] = selected_city
if selected_cuisine != st.session_state.user_input['cuisine']:
    st.session_state.user_input["cuisine"] = selected_cuisine
if rating_preference != st.session_state.user_input['rating']:
    st.session_state.user_input["rating"] = rating_preference
if cost_preference != st.session_state.user_input['cost']:
    st.session_state.user_input["cost"] = cost_preference
if rating_count_preference != st.session_state.user_input['rating_count']:
    st.session_state.user_input["rating_count"] = rating_count_preference

user_input = st.session_state.user_input

if st.sidebar.button("Get Recommendations"):
    input_df = pd.DataFrame([user_input])

    # One-Hot Encoding for User Input (using pre-trained encoder)
    encoded_data = []
    one = OneHotEncoder()
    encoder_col = one.fit_transform(input_df[['city', 'cuisine']])
    encoded_data = pd.DataFrame(encoder_col.toarray(), columns=(['City', 'Cuisine']))
    concat_df = pd.concat([encoded_data, input_df], axis=1)
    concat_df.drop(columns=['city', 'cuisine'], inplace=True)
    st.write(concat_df)
    # # Predict the cluster of the input data
    input_cluster = kmeans.predict(concat_df)[0]

    # # Ensure cluster assignments for the cleaned dataset
    # if 'cluster' not in cleaned_df.columns:
    #     cleaned_df["cluster"] = kmeans.predict(encoded_df)

    # # Filter recommendations
    # recommended_restaurants = cleaned_df[(cleaned_df["cluster"] == input_cluster) &
    #                                     (cleaned_df["rating"] >= user_input["rating"]) &
    #                                     (cleaned_df["cost"] <= user_input["cost"]) &
    #                                     (cleaned_df["rating_count"] >= user_input["rating_count"])]

    # # Display results
    # if recommended_restaurants.empty:
    #     st.write("No restaurants found matching your preferences.")
    # else:
    #     st.write(f"Recommended Restaurants from Cluster {input_cluster}:")
    #     st.write(recommended_restaurants[["restaurant_name", "city", "cuisine", "rating", "cost", "rating_count"]])
