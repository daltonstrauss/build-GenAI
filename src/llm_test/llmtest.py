import openai
import os

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key 
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
