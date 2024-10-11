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

> A Streamlit-based application that recommends London areas based on user preferences and requirements.

**Live App:** [London Borough Recommender](https://londonborough.streamlit.app/)

**Documentation:** [Process of building it](https://londonborough.streamlit.app/)

## Project Overview

The London Area Recommender System is designed to help users find suitable areas to live in London based on their personal preferences and requirements. By analyzing various factors such as housing prices, crime rates, transportation accessibility, and amenities, this application provides tailored recommendations for potential residents of London.

## Installation and Setup

To set up the project on your local machine, follow the instructions below:

### Codes and Resources Used
- **Editor Used:** Visual Studio Code
- **Python Version:** 3.8 (or later)

### Python Packages Used

#### General Purpose
- `streamlit`
- `pandas`
- `numpy`

#### Data Acquisition
- `requests`

#### Data Processing
- `scikit-learn`

#### Data Visualization
- `plotly.express`
- `folium`

## Data

### Source Data
- **London Datastore:** Provides various datasets about London boroughs, including housing prices, crime statistics, and transportation data.
- **OpenStreetMap:** Used for geospatial data and mapping features.

### Data Acquisition
The London data is fetched using the `requests` library and various APIs. The data acquisition process is handled in separate scripts within the `data_acquisition` directory.

### Data Preprocessing
The acquired data is preprocessed, including cleaning, normalization, and feature engineering, in the `data_processing` directory. This ensures that the data is ready for analysis and recommendation generation.

## Code Structure

The project is organized as follows:

```bash
├── app.py
├── data_acquisition
│   ├── fetch_housing_data.py
│   ├── fetch_crime_data.py
│   └── fetch_transport_data.py
├── data_processing
│   ├── preprocess_data.py
│   └── feature_engineering.py
├── models
│   └── recommender_model.py
├── utils
│   ├── data_loader.py
│   └── visualization.py
├── tests
│   ├── test_data_acquisition.py
│   ├── test_data_processing.py
│   └── test_recommender.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## Features

- User-friendly interface for inputting preferences and requirements
- Interactive map visualization of recommended areas
- Detailed information about each recommended area, including housing prices, crime rates, and nearby amenities
- Customizable weighting of different factors for personalized recommendations

## How to Use

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Input your preferences and requirements in the sidebar
5. Explore the recommended areas on the interactive map and in the detailed results section

## Contributing

Contributions to improve the London Area Recommender System are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
