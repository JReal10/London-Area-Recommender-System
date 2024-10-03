# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

#geopandas
import geopandas as gpd
import geodatasets as gds
from shapely import wkt

import streamlit as st
import os

import warnings
warnings.filterwarnings('ignore')

# %%
def generate_user_preference(dataframe):
    
    """
    Generate random user preferences based on the range of numeric columns in the DataFrame for a single user.
    
    Args:
    dataframe (pd.DataFrame): The input DataFrame.

    Returns:
    np.ndarray: An array of random user preferences.
    """
    
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    min_values = dataframe[numeric_columns].min()
    max_values = dataframe[numeric_columns].max()
    user_preference = np.random.rand(1, len(numeric_columns)) * (max_values - min_values).values + min_values.values
    
    print(f"User Preferences: {np.round(user_preference, 2)}")
    
    return user_preference


# %%
def visualize_data(data):
    """
    Visualizes the given data by plotting the 'Cluster' column on a map.

    Parameters:
    data (GeoDataFrame): A GeoDataFrame containing the data to be visualized. 
                         It should have a 'Cluster' column for plotting.

    Returns:
    None: This function does not return any value. It displays a plot.
    """
    # Create a map that plots the column 'Cluster'
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    data.plot(column='Cluster', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    plt.title('London Borough Clusters')
    plt.show()



# %%
def ranked_borough(dataframe, clusters, user_preference, scaler, pca, kmeans):
    """
    Recommend boroughs based on a single user's preferences and clustered data.
    
    Args:
    dataframe (pd.DataFrame): The input DataFrame.
    clusters (np.ndarray): Cluster labels for the data.
    user_preference (np.ndarray): Single user preferences to match with clusters.
    scaler (StandardScaler): The scaler used for data standardization.
    pca (PCA): The PCA model used for dimensionality reduction.
    kmeans (KMeans): The KMeans model used for clustering.

    Returns:
    list: Recommended boroughs for the nearest user cluster.
    """
    
    user_preference_scaled = scaler.transform(user_preference)
    user_pca_preference = pca.transform(user_preference_scaled)
    centroids = kmeans.cluster_centers_
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(centroids)
    distances, nearest_cluster_indices = nn.kneighbors(user_pca_preference)
    nearest_cluster = nearest_cluster_indices.flatten()[0]
    print("Nearest Cluster for User Preference:", nearest_cluster)
    
    dataframe['Cluster'] = clusters
    nearest_cluster_boroughs = dataframe[dataframe['Cluster'] == nearest_cluster]
    
    # Calculate distances between user preferences and boroughs in the nearest cluster
    cluster_data = nearest_cluster_boroughs.drop(['Area', 'Cluster'], axis=1)
    cluster_data_scaled = scaler.transform(cluster_data)
    cluster_data_pca = pca.transform(cluster_data_scaled)
    distances = cdist(user_pca_preference, cluster_data_pca, 'euclidean').flatten()
    
    nearest_cluster_boroughs['Distance'] = distances
    ranked_boroughs = nearest_cluster_boroughs.sort_values(by='Distance')
    ranked_boroughs = ranked_boroughs[['Area', 'Distance']]
    
    print("Ranked Boroughs based on User Preference:")
    print(ranked_boroughs)
    return ranked_boroughs

# %%
# Main execution

folder_dir = os.path.abspath(os.getcwd())
file_path = os.path.join(folder_dir, "London-Area-Recommender-System\\data\\transformed_data\\london_borough.csv")

df = pd.read_csv(file_path)
df = gpd.GeoDataFrame(df)
df['geometry'] = df['geometry'].apply(wkt.loads)

# Assuming your geometry column is named 'geometry'
df = df.set_geometry('geometry')
#ranked_boroughs = ranked_borough(imputed_df, clusters, user_preference, scaler, pca, kmeans)


# Streamlit UI
st.title("London Borough Recommender")

st.sidebar.header("User Preferences")

importance_levels = {
    "Not important": 1,
    "Slightly important": 2,
    "Moderately important": 3,
    "Important": 4,
    "Very important": 5
}

user_preferences = {}
for col in df.columns:
    if col != 'Area':
        user_preferences[col] = st.sidebar.radio(
            f"How important is {col} to you?",
            options=list(importance_levels.keys())
        )

if st.sidebar.button("Submit"):
    user_preference = np.array([[importance_levels[user_preferences[col]] for col in user_preferences]])
    
    # Assuming you have already performed clustering and have the 'Cluster' column in df
    # Create a map that plots the column 'Cluster'
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df.plot(column='Cluster', cmap='viridis', linewidth=0.8, ax=ax, legend=True)
    plt.title('London Borough Clusters')
    st.pyplot(fig)
else:
    st.write("Please enter your preferences and click Submit.")