import openai
import os


openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key 
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

from langchain.chat_models import ChatOpenAI # this is the new import statement
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from typing import List
from random import sample 
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./tv-reviews.csv')
data = loader.load()

print(data)