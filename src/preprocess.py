"""
Data preprocessing module for retail review sentiment analysis pipeline.
Handles cleaning, tokenization, and preparation of review text data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Clean text by removing HTML tags, special characters, and extra whitespace.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_lemmatize(text):
    """
    Tokenize text and lemmatize tokens, removing stopwords.
    
    Args:
        text (str): Cleaned text to tokenize
        
    Returns:
        str: Space-separated lemmatized tokens
    """
    if not text:
        return ""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    lemmatized_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    return ' '.join(lemmatized_tokens)


def preprocess_reviews(input_file='data/raw_reviews.csv', output_file='data/processed_reviews.csv'):
    """
    Main preprocessing function that reads raw reviews and outputs cleaned data.
    
    Args:
        input_file (str): Path to raw reviews CSV file
        output_file (str): Path to save processed reviews CSV file
    """
    print("Loading raw reviews...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure the file exists.")
        return
    
    print(f"Loaded {len(df)} reviews")
    
    # Handle missing values
    print("Handling missing values...")
    df = df.dropna(subset=['review_text'])
    
    # Remove duplicates
    print("Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['review_text'])
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate reviews")
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Remove empty reviews after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Tokenize and lemmatize
    print("Tokenizing and lemmatizing...")
    df['processed_text'] = df['cleaned_text'].apply(tokenize_and_lemmatize)
    
    # Remove reviews with no tokens after processing
    df = df[df['processed_text'].str.len() > 0]
    
    # Save processed data
    print(f"Saving processed reviews to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete! Processed {len(df)} reviews saved to {output_file}")


if __name__ == "__main__":
    preprocess_reviews()


# Text cleaning function added

# Lemmatization added

# Error handling improved

# Progress indicators added

# Updated: 2025-12-05T18:46:39.965563

# Updated: 2025-12-05T18:46:40.280514

# Updated: 2025-12-05T18:46:40.586039

# Updated: 2025-12-05T18:46:42.064460
