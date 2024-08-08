# Reddit Stock Sentiment Analyzer

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/JReal10/Reddit-Stock-Sentiment-Analyzer?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/JReal10/Reddit-Stock-Sentiment-Analyzer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/JReal10/Reddit-Stock-Sentiment-Analyzer)
![GitHub](https://img.shields.io/github/license/JReal10/Reddit-Stock-Sentiment-Analyzer)
![contributors](https://img.shields.io/github/contributors/JReal10/Reddit-Stock-Sentiment-Analyzer)
![codesize](https://img.shields.io/github/languages/code-size/JReal10/Reddit-Stock-Sentiment-Analyzer)

> A Streamlit-based application that analyzes the sentiment of Reddit comments related to a given stock.

## Project Overview

The Reddit Stock Sentiment Analyzer project aims to provide a comprehensive analysis of the sentiment surrounding a specific stock based on comments from the Reddit community. By leveraging natural language processing techniques, this application extracts and visualizes the sentiment of Reddit discussions, allowing users to gain insights into the overall market sentiment for a particular stock.

## Installation and Setup

To set up the project on your local machine, follow the instructions below:

### Codes and Resources Used

- **Editor Used:** Visual Studio Code
- **Python Version:** 3.8

### Python Packages Used

#### General Purpose

- `streamlit`
- `pandas`
- `numpy`
- `re`

#### Data Acquisition
- `requests`

#### Data Processing
- `transformers`
- `torch`

#### Data Visualization
- `plotly.express`

## Data

### Source Data

- **Reddit API:** Fetches comments from the "stocks" and "wallstreetbets" subreddits.

### Data Acquisition

The Reddit data is fetched using the `requests` library and the Reddit API. The data acquisition process is handled in the `fetch_reddit_data()` function.

### Data Preprocessing

The Reddit comments are preprocessed, including text cleaning and sentiment analysis, in the `process_reddit_data()` function. The sentiment analysis is performed using a pre-trained transformer model from the `transformers` library.

## Code Structure

The project is organized as follows:

```bash
├── app.py
├── models
│   └── transformer_model.py
├── scripts
│   ├── fetch_reddit_data.py
│   ├── process_reddit_data.py
│   └── database_manager.py
├── LICENSE
├── README.md
└── .gitignore
