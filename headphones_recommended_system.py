import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# Load your dataset (replace 'headphones.csv' with your actual file path)
df = pd.read_csv('headphones.csv')

# Select relevant features for recommendation
features = ['Brand', 'Form_Factor', 'Connectivity_Type', 'Colour']

# One-Hot Encoding categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[features].fillna('Unknown')).toarray()

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(encoded_features)

# Streamlit app
st.title("🎧 Headphone Recommendation System")

# Sidebar for user inputs
st.sidebar.header("User Preferences")

# Get unique values for selection
brands = ['Any'] + sorted(df['Brand'].dropna().unique().tolist())
form_factors = sorted(df['Form_Factor'].dropna().unique().tolist())
connectivity_types = sorted(df['Connectivity_Type'].dropna().unique().tolist())
colors = ['Any'] + sorted(df['Colour'].dropna().unique().tolist())

# User inputs
selected_brand = st.sidebar.selectbox("Preferred Brand:", brands)
selected_form_factor = st.sidebar.selectbox("Form Factor:", form_factors)
selected_connectivity = st.sidebar.selectbox("Connectivity Type:", connectivity_types)
price_min, price_max = st.sidebar.slider(
    "Price Range (Rs):",
    int(df['Actual_Price'].min()),
    int(df['Actual_Price'].max()),
    (int(df['Actual_Price'].min()), int(df['Actual_Price'].max()))
)
selected_color = st.sidebar.selectbox("Preferred Color:", colors)

# Filter based on user preferences
filtered_df = df.copy()

if selected_brand != 'Any':
    filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]

filtered_df = filtered_df[filtered_df['Form_Factor'] == selected_form_factor]
filtered_df = filtered_df[filtered_df['Connectivity_Type'] == selected_connectivity]
filtered_df = filtered_df[(filtered_df['Actual_Price'] >= price_min) & (filtered_df['Actual_Price'] <= price_max)]

if selected_color != 'Any':
    filtered_df = filtered_df[filtered_df['Colour'] == selected_color]

if filtered_df.empty:
    st.write(" No headphones match your preferences. Please adjust your filters.")
else:
    st.write("## Matching Headphones:")
    for index, row in filtered_df.iterrows():
        st.write(f"**{row['Title']}** - Rs{row['Actual_Price']}")

    # Let the user select a headphone for recommendations
    selected_headphone = st.selectbox(
        "Select a headphone to see similar recommendations:",
        filtered_df['Title'].tolist()
    )

    # Get index of selected headphone
    selected_index = df[df['Title'] == selected_headphone].index[0]

    # Filter the indices and features for the filtered DataFrame
    filtered_indices = filtered_df.index
    filtered_encoded = encoded_features[filtered_indices]

    # Calculate similarity for the filtered headphones
    selected_encoded = encoded_features[selected_index].reshape(1, -1)
    filtered_similarity_matrix = cosine_similarity(selected_encoded, filtered_encoded)

    # Get similarity scores for the selected item
    similarity_scores = list(enumerate(filtered_similarity_matrix[0]))

    # Sort items based on similarity scores
    sorted_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N recommendations (excluding the selected item)
    num_recommendations = 5
    recommendations = []
    for item in sorted_items:
        idx = filtered_indices[item[0]]
        if df['Title'].iloc[idx] != selected_headphone:
            recommendations.append((idx, item[1]))
        if len(recommendations) == num_recommendations:
            break

    st.write(f"##  Recommendations similar to **{selected_headphone}**:")
    for i, (index, score) in enumerate(recommendations):
        headphone = df.iloc[index]
        st.write(
            f"{i+1}. **{headphone['Title']}** - Rs{headphone['Actual_Price']} "
            f"(Similarity Score: {score:.2f})"
        )
