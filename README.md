# Retail Review Sentiment & Insights Pipeline

## Project Overview

This project builds an end-to-end NLP pipeline to analyze customer reviews from e-commerce platforms and extract actionable insights for consumer behavior and product improvement.

The pipeline performs:

- **Sentiment Analysis**: classifying reviews as positive, negative, or neutral.
- **Topic Modeling**: identifying recurring themes and customer concerns.
- **Trend Analysis**: tracking sentiment changes over time.
- **Visualization**: producing charts and dashboards for actionable insights.

## Dataset

Use publicly available datasets: e.g., Amazon product reviews, Yelp business reviews, Kaggle review datasets.

Store raw data in `data/raw_reviews.csv` (can include sample data).

**Note**: This repo uses open datasets for educational purposes.

## Pipeline Steps / Methodology

### Data Cleaning & Preprocessing

- Remove HTML, special characters, and stopwords.
- Lemmatize/tokenize text.
- Handle missing or duplicate entries.

### Sentiment Analysis

- **Baseline**: VADER or TextBlob for polarity scores.
- **Advanced**: Fine-tuned BERT model or HuggingFace transformers for more accurate sentiment classification.
- **Output**: sentiment score per review in `results/sentiment_scores.csv`.

### Topic Modeling

- Use LDA (Latent Dirichlet Allocation) or NMF to detect recurring themes in reviews.
- Output top keywords per topic in `results/topic_keywords.csv`.
- Optional: word clouds for visual representation.

### Trend Analysis & Visualization

- Aggregate sentiment over time to detect shifts in customer perception.
- Create visualizations: sentiment distribution, topic heatmaps, trend lines.
- Save plots in `results/visuals/`.

### Insights Generation

- Summarize key patterns: common complaints, praised features, sentiment shifts.
- Suggest actionable recommendations: e.g., improve packaging, highlight popular features, etc.

## Technical Stack

- **Python, Pandas, Numpy**
- **NLP**: NLTK, SpaCy, TextBlob, VADER, or HuggingFace Transformers
- **Topic Modeling**: Gensim, scikit-learn
- **Visualization**: Matplotlib, Seaborn, WordCloud, Plotly
- **Optional**: Jupyter Notebooks for step-by-step reproducibility

## Usage / How to Run

### Clone repo

```bash
git clone https://github.com/Krish-B-Shah/Retail-Review-Sentiment-Pipeline.git
cd Retail-Review-Sentiment-Pipeline
pip install -r requirements.txt
```

### Run Pipeline

1. **Preprocess data**: `python src/preprocess.py`
2. **Run sentiment analysis**: `python src/sentiment.py`
3. **Run topic modeling**: `python src/topic_model.py`
4. **Open notebooks** in `notebooks/` for interactive exploration and visualization.

## Expected Outcomes

- CSV of sentiment scores per review
- CSV of topic keywords per topic
- Visualizations:
  - `sentiment_distribution.png` → shows proportion of positive, negative, neutral reviews
  - `topic_wordcloud.png` → highlights recurring themes
  - `trend_over_time.png` → shows how sentiment changes over time

## Impact Statement (for NRF / Resume)

- Shows ability to analyze large-scale consumer data
- Extracts actionable insights to improve product or service
- Demonstrates transferable ML/NLP skills in a consumer/retail context
- Can be showcased in poster or demo for visual storytelling

## Optional Enhancements for NRF Shine

- Interactive dashboards using Streamlit or Plotly Dash
- Comparison of sentiment across multiple product categories
- Highlight top positive vs negative themes

## Repository Structure

```
Retail-Review-Sentiment-Pipeline/
│
├─ README.md
├─ data/
│   ├─ raw_reviews.csv          # original dataset (Amazon/Yelp/Kaggle)
│   └─ processed_reviews.csv    # cleaned/preprocessed
├─ notebooks/
│   ├─ data_cleaning.ipynb
│   ├─ sentiment_analysis.ipynb
│   ├─ topic_modeling.ipynb
│   └─ insights_visualization.ipynb
├─ src/
│   ├─ preprocess.py
│   ├─ sentiment.py
│   └─ topic_model.py
├─ results/
│   ├─ sentiment_scores.csv
│   ├─ topic_keywords.csv
│   └─ visuals/
│       ├─ sentiment_distribution.png
│       ├─ topic_wordcloud.png
│       └─ trend_over_time.png
└─ requirements.txt
```

