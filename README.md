![Header Image](https://github.com/pragyy/datascience-readme-template/blob/main/Headerheader.jpg)

# London Area Recommender

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/JReal10/London-Area-Recommender-System?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/JReal10/London-Area-Recommender-System)
![GitHub pull requests](https://img.shields.io/github/issues-pr/JReal10/London-Area-Recommender-System)
![GitHub](https://img.shields.io/github/license/JReal10/London-Area-Recommender-System)
![contributors](https://img.shields.io/github/contributors/JReal10/London-Area-Recommender-System) 
![codesize](https://img.shields.io/github/languages/code-size/JReal10/London-Area-Recommender-System) 

> A guide to writing an amazing readme for your data science project.

# Project Overview

The London Area Recommender project aims to provide personalized area recommendations within London based on user preferences and data analysis. This project utilizes various data sources and machine learning techniques to analyze and predict the best areas for individuals based on criteria such as cost of living, safety, amenities, and more.

# Installation and Setup

To set up the project on your local machine, follow the instructions below:

## Codes and Resources Used
- **Editor Used:** Visual Studio Code
- **Python Version:** 3.8

## Python Packages Used
### General Purpose
- `urllib`
- `os`
- `request`

### Data Manipulation
- `pandas`
- `numpy`

### Data Visualization
- `seaborn`
- `matplotlib`

### Machine Learning
- `scikit-learn`
- `tensorflow`

# Data

## Source Data
- **OpenStreetMap Data:** [OpenStreetMap](https://www.openstreetmap.org)
  - Description: Provides detailed geographical data for London.
- **UK Police Data:** [UK Police Data](https://data.police.uk)
  - Description: Crime statistics and safety data for different areas in London.
- **London Housing Dataset:** [Kaggle Dataset](https://www.kaggle.com)
  - Description: Housing prices and rental data for London.

## Data Acquisition
Data is collected through API calls and online scraping from the sources mentioned above. Detailed scripts for data acquisition can be found in `data_acquisition.py`.

## Data Preprocessing
Data preprocessing includes cleaning, normalization, and transformation steps to prepare the datasets for analysis. The preprocessing steps are detailed in `data_preprocessing.ipynb`.

# Code Structure
The project is organized as follows:

```bash
├── data
│   ├── raw
│   │   ├── data1.csv
│   │   ├── data2.csv
│   ├── cleaned
│   │   ├── cleaneddata1.csv
│   │   └── cleaneddata2.csv
├── scripts
│   ├── data_acquisition.py
│   ├── data_preprocessing.ipynb
│   ├── data_analysis.ipynb
│   ├── data_modelling.ipynb
├── img
│   ├── img1.png
│   ├── Headerheader.jpg
├── LICENSE
├── README.md
└── .gitignore
```

# Results and Evaluation
The results of the project include various metrics and visualizations that demonstrate the effectiveness of the area recommendations. Key findings and evaluation methodologies are detailed in data_analysis.ipynb.

## Future Work
Future improvements can include:

Incorporating real-time data updates for more accurate recommendations.
Expanding the recommendation criteria to include more user preferences.
Developing a user-friendly web interface for easier access to recommendations.

# Acknowledgments/References

Image by rashadashurov
Data sources: OpenStreetMap, UK Police Data, Kaggle

# License
This project is licensed under the MIT License.

You can access the project through: London Area Recommender System
