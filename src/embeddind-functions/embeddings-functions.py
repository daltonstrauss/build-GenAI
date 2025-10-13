import numpy as np
import sentence_transformers
from typing import Union
import openai
import os
openai.api_base = "https://openai.vocareum.com/v1"

# Define your OpenAI API key 
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2')

emb1, emb2 = model.encode(['index,season no.,episode no.,episode name,name,line\n', "0,1,1,Pilot,Rick,Morty! You gotta come on. Jus'... you gotta come with me.\n", '1,1,1,Pilot,Morty,"What, Rick? What’s going on?"\n']
)

np.allclose(emb1, emb2)

def read_quotes() -> list[str]:
    with open("rick_and_morty_quotes.csv", "r") as fh:
        return fh.readlines()
    
rick_and_morty_quotes = read_quotes()
print(rick_and_morty_quotes[:3])