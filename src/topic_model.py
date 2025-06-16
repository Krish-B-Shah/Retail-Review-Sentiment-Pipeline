"""
Topic modeling module for retail review sentiment analysis pipeline.
Uses LDA (Latent Dirichlet Allocation) to identify recurring themes.
"""

import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import re


def prepare_corpus(texts):
    """
    Prepare corpus for LDA modeling using Gensim.
    
    Args:
        texts (list): List of processed text strings
        
    Returns:
        tuple: (dictionary, corpus)
    """
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts if text and len(text) > 0]
    
    # Create dictionary
    dictionary = corpora.Dictionary(tokenized_texts)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    return dictionary, corpus


def get_topic_keywords_lda(model, dictionary, num_words=10):
    """
    Extract top keywords for each topic from LDA model.
    
    Args:
        model: Trained LDA model
        dictionary: Gensim dictionary
        num_words (int): Number of top words per topic
        
    Returns:
        list: List of dictionaries with topic keywords
    """
    topics = []
    for idx, topic in model.print_topics(-1, num_words=num_words):
        # Parse topic string to extract words and weights
        words = re.findall(r'"([^"]+)"', topic)
        weights = re.findall(r'(\d+\.\d+)', topic)
        
        topic_dict = {
            'topic_id': idx,
            'keywords': ', '.join(words[:num_words]),
            'top_words': words[:num_words]
        }
        topics.append(topic_dict)
    
    return topics


def topic_modeling_lda(input_file='data/processed_reviews.csv', output_file='results/topic_keywords.csv', 
                       num_topics=5, num_words=10):
    """
    Perform topic modeling using LDA.
    
    Args:
        input_file (str): Path to processed reviews CSV file
        output_file (str): Path to save topic keywords CSV file
        num_topics (int): Number of topics to extract
        num_words (int): Number of top words per topic
    """
    print("Loading processed reviews...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run preprocessing first.")
        return
    
    # Filter out empty processed texts
    df = df[df['processed_text'].notna() & (df['processed_text'].str.len() > 0)]
    
    print(f"Performing topic modeling on {len(df)} reviews...")
    print(f"Extracting {num_topics} topics...")
    
    # Prepare corpus
    texts = df['processed_text'].tolist()
    dictionary, corpus = prepare_corpus(texts)
    
    # Train LDA model
    print("Training LDA model...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Extract topic keywords
    print("Extracting topic keywords...")
    topics = get_topic_keywords_lda(lda_model, dictionary, num_words)
    
    # Create output DataFrame
    topic_df = pd.DataFrame(topics)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    print(f"Saving topic keywords to {output_file}...")
    topic_df.to_csv(output_file, index=False)
    print(f"Topic modeling complete! Results saved to {output_file}")
    
    # Print topics
    print("\nExtracted Topics:")
    for _, row in topic_df.iterrows():
        print(f"\nTopic {row['topic_id']}: {row['keywords']}")


def topic_modeling_sklearn(input_file='data/processed_reviews.csv', output_file='results/topic_keywords.csv',
                           num_topics=5, num_words=10):
    """
    Alternative topic modeling using scikit-learn's LDA.
    
    Args:
        input_file (str): Path to processed reviews CSV file
        output_file (str): Path to save topic keywords CSV file
        num_topics (int): Number of topics to extract
        num_words (int): Number of top words per topic
    """
    print("Loading processed reviews...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run preprocessing first.")
        return
    
    # Filter out empty processed texts
    df = df[df['processed_text'].notna() & (df['processed_text'].str.len() > 0)]
    
    print(f"Performing topic modeling on {len(df)} reviews...")
    print(f"Extracting {num_topics} topics...")
    
    # Prepare data
    texts = df['processed_text'].tolist()
    
    # Create count vectorizer
    vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.5)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    print("Training LDA model...")
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10
    )
    lda_model.fit(doc_term_matrix)
    
    # Extract topic keywords
    print("Extracting topic keywords...")
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        topic_dict = {
            'topic_id': topic_idx,
            'keywords': ', '.join(top_words),
            'top_words': top_words
        }
        topics.append(topic_dict)
    
    # Create output DataFrame
    topic_df = pd.DataFrame(topics)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    print(f"Saving topic keywords to {output_file}...")
    topic_df.to_csv(output_file, index=False)
    print(f"Topic modeling complete! Results saved to {output_file}")
    
    # Print topics
    print("\nExtracted Topics:")
    for _, row in topic_df.iterrows():
        print(f"\nTopic {row['topic_id']}: {row['keywords']}")


if __name__ == "__main__":
    # Use Gensim LDA by default
    topic_modeling_lda(num_topics=5, num_words=10)


# LDA implementation

# Parameters optimized

# Sklearn alternative added

# Update 2025-12-05 18:46:26.612413

# Update 2025-12-05 18:46:27.814636

# Update 2025-12-05 18:46:28.757990

# Update 2025-12-05 18:46:28.907199

# Update 2025-12-05 18:46:30.018664

# Update 2025-12-05 18:46:30.204223

# Updated: 2025-06-16T10:25:16
