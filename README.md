<p align="center">
  <img src="thumbnail/LondonImage.jpg" alt="Header Image" width="100%" height="450px">
</p>

# London Borough Recommender
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/JReal10/London-Area-Recommender-System?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/JReal10/London-Area-Recommender-System)
![GitHub pull requests](https://img.shields.io/github/issues-pr/JReal10/London-Area-Recommender-System)
![GitHub](https://img.shields.io/github/license/JReal10/London-Area-Recommender-System)
![contributors](https://img.shields.io/github/contributors/JReal10/London-Area-Recommender-System)
![codesize](https://img.shields.io/github/languages/code-size/JReal10/London-Area-Recommender-System)

> A Streamlit-based application that recommends London boroughs based on user preferences and socio-economic data.

**Live App:** [London Borough Recommender](https://londonborough.streamlit.app/)

**Documentation:** [Process of building it](https://github.com/JReal10/London-Area-Recommender-System/blob/main/area_recommender.ipynb)

## Project Overview

The London Borough Recommender is a data-driven system that helps users discover suitable areas to live in London based on their personal preferences. It analyzes socio-economic factors across boroughs, such as housing prices, crime rates, unemployment rates, education, and environmental factors, to provide tailored recommendations.

This project combines machine learning techniques like clustering and dimensionality reduction to classify boroughs based on similar characteristics and suggests boroughs that align with user preferences.

## Installation and Setup

### Pre-requisites

To set up the project on your local machine, follow the instructions below:

- **Editor:** Visual Studio Code (or any preferred IDE)
- **Python Version:** 3.8 or later

### Required Python Libraries
The key Python packages used in the project are:

- **General Purpose:**
  - `streamlit`
  - `pandas`
  - `numpy`

- **Geospatial Processing:**
  - `geopandas`
  - `shapely`

- **Data Processing & Machine Learning:**
  - `scikit-learn` (for clustering and dimensionality reduction)
  - `matplotlib` (for data visualization)
  - `seaborn`
  - `scipy`

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/JReal10/London-Area-Recommender-System.git
   ```

2. Navigate to the project directory:
   ```bash
   cd London-Area-Recommender-System
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Data

### Data Sources
The project integrates socio-economic data from multiple sources:

- **London Datastore:** Provides borough-level data on housing, crime, education, unemployment, and environmental factors.
- **Trust for London:** Provides borough-level data on poverty rates, unemployment, and education achievements. 
  - **Poverty rate 2022/2023:** Percentage of the population living under the poverty line. [Source](https://trustforlondon.org.uk/)
  - **Unemployment rate:** Percentage of unemployed individuals in the area. [Source](https://trustforlondon.org.uk/)
  - **Education Achievement:** Percentage of pupils who achieved grade 9-4 in secondary education. [Source](https://trustforlondon.org.uk/)
- **Statistical GIS Boundaries (OpenStreetMap):** Provides geospatial data for mapping the London boroughs.


### Data Preprocessing
The raw data is processed through the following steps:

- **Loading and Cleaning:** Socio-economic and geographical data are cleaned and merged to ensure consistency in borough names and structures.
- **Normalization:** Data is standardized to ensure all features (e.g., crime rates, property prices) are on the same scale.
- **Dimensionality Reduction:** Principal Component Analysis (PCA) is used to reduce the data's complexity while retaining variance for clustering.
- **Clustering:** K-Means clustering is applied to group similar boroughs based on their characteristics.

## Visualization
The project provides multiple data visualizations, including:

- **Pairplots and heatmaps** for correlation analysis.
- **Scatter plots** of PCA components to visualize borough clusters.
- **Choropleth maps** for geospatial visualization of the clustered boroughs.

## Code Structure
The project is organized into several directories for better modularity:

```bash
Area_Recommender
├─ .vscode
│  └─ settings.json                    # Settings for Visual Studio Code
├─ London-Area-Recommender-System
│  ├─ .devcontainer
│  │  └─ devcontainer.json              # Dev container configuration for VS Code
│  ├─ app.py                            # Main Streamlit application
│  ├─ area_recommender.ipynb            # Jupyter notebook for development and documentation
│  ├─ data                              # Data directory
│  │  ├─ london_borough_metadata.csv    # Metadata for London boroughs
│  │  ├─ raw_data                       # Raw data sources
│  │  │  └─ London-wards-2018           # Shapefiles and raw geographic data
│  │  └─ transformed_data
│  │     └─ london_borough.csv          # Transformed data ready for analysis
│  ├─ LICENSE
│  ├─ README.md
│  ├─ requirements.txt                  # Dependencies
│  └─ thumbnail
     └─ LondonImage.jpg                # Thumbnail image for README
```


## Features

- **User-friendly Interface:** Simple inputs for users to specify preferences (e.g., housing prices, crime rates).
- **Borough Clustering:** London boroughs are grouped based on socio-economic similarities.
- **Interactive Map:** View boroughs and their cluster assignments on a choropleth map.
- **Customizable Recommendations:** Users can adjust the importance of factors such as property prices or crime rates to receive personalized recommendations.

## How to Use

1. Input your preferences for various socio-economic factors (e.g., budget, safety, schools).
2. The recommender system clusters and ranks boroughs that best align with your preferences.
3. Explore borough recommendations on the interactive map and view detailed information about each recommended area.

### Example Usage

Once the app is running, you can:

- Adjust the sliders on the sidebar to input your preferences.
- View the recommended boroughs highlighted on the map.
- Explore each recommended borough based on its socio-economic factors.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

vbnet
Copy code

This is the complete markdown file, which you can directly integrate into your project's root directo
