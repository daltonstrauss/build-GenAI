import pandas as pd
import json
import openai
import os
from langchain.chat_models import ChatOpenAI # this is the new import statement
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from typing import List
from random import sample 
from langchain.document_loaders.csv_loader import CSVLoader



openai.api_base = "https://openai.vocareum.com/v1"
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key



def load_and_clean_csv(csv_file):
    """
    Load and clean the TV reviews CSV file.
    """
    # Read CSV without header since first row contains data
    df = pd.read_csv(csv_file, header=None)
    
    # Set proper column names
    df.columns = ['TV_Name', 'Review_Title', 'Review_Rating', 'Review_Text']
    
    # Clean the data by removing prefixes
    df['TV_Name'] = df['TV_Name'].str.replace('TV Name: ', '', regex=False)
    df['Review_Title'] = df['Review_Title'].str.replace('Review Title: ', '', regex=False)
    df['Review_Rating'] = df['Review_Rating'].str.replace('Review Rating: ', '', regex=False)
    df['Review_Text'] = df['Review_Text'].str.replace('Review Text: ', '', regex=False)
    
    # Convert rating to numeric
    df['Review_Rating'] = pd.to_numeric(df['Review_Rating'])
    
    return df

def classify_reviews_by_rating(df, rating_threshold=6):
    """
    Classify reviews as positive/negative based on rating threshold.
    
    Args:
        df: DataFrame with cleaned review data
        rating_threshold: Reviews with rating > threshold are positive
    
    Returns:
        Dictionary with positive and negative indices
    """
    positive_indices = []
    negative_indices = []
    
    for index, row in df.iterrows():
        if row['Review_Rating'] > rating_threshold:
            positive_indices.append(index)
        else:
            negative_indices.append(index)
    
    return {
        "positives": positive_indices,
        "negatives": negative_indices
    }

def main():
    # Load and process the data
    csv_file = 'tv_reviews.csv'
    df = load_and_clean_csv(csv_file)
    
    print(f"Loaded {len(df)} reviews")
    print(f"Rating distribution:\n{df['Review_Rating'].value_counts().sort_index()}")
    
    # Classify reviews (you can adjust the threshold as needed)
    sentiment_result = classify_reviews_by_rating(df, rating_threshold=6)
    
    # Convert to JSON format
    result_json = json.dumps(sentiment_result)
    
    print("\n=== Final JSON Output ===")
    print(result_json)
    
    # Pretty print for readability
    print("\n=== Pretty Formatted JSON ===")
    print(json.dumps(sentiment_result, indent=2))
    
    # Save to file
    with open('sentiment_output.json', 'w') as f:
        f.write(result_json)
    
    print("\nJSON saved to 'sentiment_output.json'")
    
    return sentiment_result

if __name__ == "__main__":
    result = main()
