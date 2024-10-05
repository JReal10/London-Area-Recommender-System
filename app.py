# %%
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import wkt
import os

# Set page config
st.set_page_config(page_title="London Borough Recommender", page_icon="🏙️", layout="wide")

# Load data
@st.cache_data
def load_data():
    folder_dir = os.path.abspath(os.getcwd())
    file_path = os.path.join(folder_dir, "London-Area-Recommender-System", "data", "transformed_data", "london_borough.csv")
    df = pd.read_csv(file_path)
    df = gpd.GeoDataFrame(df)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = df.set_geometry('geometry')
    return df

df = load_data()

# Define cluster characteristics weights
cluster_weights = {
    'Affordability': [2, 1, 0, 3, 0],
    'Safety and low crime': [1, 3, 2, 0, 3],
    'Public transport and accessibility': [0, 3, 1, 0, 3],
    'Access to green spaces': [1, 2, 3, 0, 2]
}

education_weights = {
    'Very important': [2, 2, 1, 0, 3],
    'Moderately important': [1, 1, 1, 1, 1],
    'Not important': [0, 0, 0, 1, 0]
}

transport_weights = {
    'High': [0, 2, 1, 0, 3],
    'Moderate': [1, 1, 1, 1, 1],
    'Low, prefer less busy areas': [2, 1, 1, 2, 0]
}

environment_weights = {
    'Very concerned': [1, 1, 3, 0, 2],
    'Moderately concerned': [1, 1, 2, 1, 1],
    'Not a major concern': [0, 0, 0, 2, 0]
}

property_price_weights = {
    'High-end areas, cost is not a problem': [1, 0, 0, 0, 3],
    'Mid-range pricing with good balance': [1, 2, 1, 1, 1],
    'More affordable with potential for growth': [0, 0, 0, 3, 0]
}

# Function to calculate the scores and recommend the best cluster
def recommend_cluster(priority, education, transport, environment, property_price):
    cluster_scores = [0, 0, 0, 0, 0]
    
    for i in range(len(cluster_scores)):
        cluster_scores[i] += cluster_weights[priority][i]
        cluster_scores[i] += education_weights[education][i]
        cluster_scores[i] += transport_weights[transport][i]
        cluster_scores[i] += environment_weights[environment][i]
        cluster_scores[i] += property_price_weights[property_price][i]
    
    recommended_cluster = np.argmax(cluster_scores)
    return recommended_cluster, cluster_scores

# Streamlit UI
st.title("🏙️ London Borough Recommender")

st.markdown("""
This app helps you find the best London borough based on your preferences. 
Answer a few questions about your priorities, and we'll recommend the most suitable areas for you!
""")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Preferences")
    
    priority = st.selectbox(
        "What is your top priority?",
        ("Affordability", "Safety and low crime", "Public transport and accessibility", "Access to green spaces")
    )
    
    education = st.selectbox(
        "Importance of access to good schools?",
        ("Very important", "Moderately important", "Not important")
    )
    
    transport = st.selectbox(
        "Ideal level of public transport?",
        ("High", "Moderate", "Low, prefer less busy areas")
    )
    
    environment = st.selectbox(
        "Concern about environmental factors?",
        ("Very concerned", "Moderately concerned", "Not a major concern")
    )
    
    property_price = st.selectbox(
        "Property pricing preference?",
        ("High-end areas, cost is not a problem", "Mid-range pricing with good balance", "More affordable with potential for growth")
    )
    
    if st.button('Get Recommendation', type='primary'):
        recommended_cluster, cluster_scores = recommend_cluster(priority, education, transport, environment, property_price)
        
        df['highlight'] = np.where(df['Cluster'] == recommended_cluster, 'Recommended', 'Other')
        
        stat_df = df[df['Cluster'] == recommended_cluster].drop(columns=['geometry'])
        recommended_boroughs = stat_df['Area'].tolist()
        
        with col2:
            st.subheader("Recommendation Results")
            st.success(f"Based on your preferences, we recommend Cluster {recommended_cluster}.")
            st.write(f"**Recommended Boroughs:** {', '.join(recommended_boroughs)}")
            
            # Create a map
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            cmap = {'Recommended': '#1E88E5', 'Other': '#E0E0E0'}
            df.plot(column='highlight', color=df['highlight'].map(cmap), linewidth=0.8, edgecolor='0.8', ax=ax, legend=True)
            ax.axis('off')
            plt.title(f'London Boroughs - Highlighting Cluster {recommended_cluster}')
            st.pyplot(fig)
            
            # Display scores
            st.subheader("Category Scores")
            scores = {
                "Education": education_weights[education][recommended_cluster],
                "Transport": transport_weights[transport][recommended_cluster],
                "Environment": environment_weights[environment][recommended_cluster],
                "Property Price": property_price_weights[property_price][recommended_cluster]
            }
            
            for category, score in scores.items():
                st.metric(label=category, value=score)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")