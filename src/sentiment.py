"""
Sentiment analysis module for retail review sentiment analysis pipeline.
Uses VADER and TextBlob for sentiment classification.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def get_vader_sentiment(text):
    """
    Get sentiment scores using VADER.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment scores (compound, pos, neu, neg)
    """
    if pd.isna(text) or not text:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    return analyzer.polarity_scores(str(text))


def get_textblob_sentiment(text):
    """
    Get sentiment polarity using TextBlob.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Polarity score (-1 to 1)
    """
    if pd.isna(text) or not text:
        return 0.0
    
    blob = TextBlob(str(text))
    return blob.sentiment.polarity


def classify_sentiment(compound_score):
    """
    Classify sentiment based on compound score.
    
    Args:
        compound_score (float): VADER compound score
        
    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def analyze_sentiment(input_file='data/processed_reviews.csv', output_file='results/sentiment_scores.csv'):
    """
    Main sentiment analysis function.
    
    Args:
        input_file (str): Path to processed reviews CSV file
        output_file (str): Path to save sentiment scores CSV file
    """
    print("Loading processed reviews...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run preprocessing first.")
        return
    
    print(f"Analyzing sentiment for {len(df)} reviews...")
    
    # Get VADER sentiment scores
    print("Computing VADER sentiment scores...")
    vader_scores = df['cleaned_text'].apply(get_vader_sentiment)
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    
    # Get TextBlob sentiment scores
    print("Computing TextBlob sentiment scores...")
    df['textblob_polarity'] = df['cleaned_text'].apply(get_textblob_sentiment)
    
    # Classify sentiment using VADER compound score
    df['sentiment_label'] = df['vader_compound'].apply(classify_sentiment)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    print(f"Saving sentiment scores to {output_file}...")
    output_df = df[['review_text', 'cleaned_text', 'vader_compound', 'vader_pos', 
                    'vader_neu', 'vader_neg', 'textblob_polarity', 'sentiment_label']].copy()
    
    # Add review_id if it exists, otherwise create one
    if 'review_id' in df.columns:
        output_df['review_id'] = df['review_id']
    else:
        output_df['review_id'] = range(1, len(output_df) + 1)
    
    # Reorder columns
    cols = ['review_id', 'review_text', 'cleaned_text', 'vader_compound', 
            'vader_pos', 'vader_neu', 'vader_neg', 'textblob_polarity', 'sentiment_label']
    output_df = output_df[cols]
    
    output_df.to_csv(output_file, index=False)
    print(f"Sentiment analysis complete! Results saved to {output_file}")
    
    # Print summary statistics
    print("\nSentiment Distribution:")
    print(output_df['sentiment_label'].value_counts())
    print(f"\nAverage VADER Compound Score: {output_df['vader_compound'].mean():.3f}")
    print(f"Average TextBlob Polarity: {output_df['textblob_polarity'].mean():.3f}")


if __name__ == "__main__":
    analyze_sentiment()


# VADER integration

# TextBlob added

# Thresholds updated

# Classification added

# Update 2025-12-05 18:46:26.211421

# Update 2025-12-05 18:46:27.003388

# Update 2025-12-05 18:46:27.143525

# Update 2025-12-05 18:46:27.405657

# Update 2025-12-05 18:46:27.958922


# Updated: 2025-06-06T16:28:03
