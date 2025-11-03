import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    print("OPENAI_API_KEY not set in .env file.")
    exit()

# --- Pydantic Data Models for Structured Output ---

class Listing(BaseModel):
    neighborhood: str = Field(description="Name of the neighborhood.")
    price: int = Field(description="Price of the property in USD.")
    bedrooms: int = Field(description="Number of bedrooms.")
    bathrooms: float = Field(description="Number of bathrooms (e.g., 2.5).")
    house_size_sqft: int = Field(description="Total house size in square feet.")
    description: str = Field(description="A detailed description of the property itself.")
    neighborhood_description: str = Field(description="A description of the surrounding neighborhood, its vibe, and amenities.")

class ListingCollection(BaseModel):
    listings: List[Listing] = Field(description="A list of 10 real estate listings.")

# --- Main Generation Logic ---

def generate_listings():
    """
    Uses an LLM to generate 10 synthetic real estate listings and saves them to listings.json.
    """
    print("Initializing LLM and Output Parser...")
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Initialize the parser
    parser = PydanticOutputParser(pydantic_object=ListingCollection)
    
    # Create the prompt template
    prompt_template = """
    You are an expert real estate data generator. Create a diverse and realistic
    set of 10 property listings. Include a mix of property types (e.g., condo,
    suburban home, urban apartment), neighborhoods, prices, and features.
    
    For each listing, provide:
    1.  A plausible neighborhood name.
    2.  A price (e.g., 450000, 1200000).
    3.  Number of bedrooms.
    4.  Number of bathrooms.
    5.  House size in sqft.
    6.  A detailed property description (approx. 3-4 sentences).
    7.  A detailed neighborhood description (approx. 2-3 sentences).

    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create the chain
    chain = prompt | llm | parser
    
    print("Generating 10 listings... (This may take a moment)")
    
    try:
        # Invoke the chain
        result = chain.invoke({})
        
        # Save the listings to a JSON file
        with open("listings.json", "w") as f:
            # Use .model_dump_json() for Pydantic v2
            json.dump(result.model_dump(), f, indent=2)
            
        print("\n✅ Successfully generated and saved 10 listings to 'listings.json'.")
        print(f"Example listing neighborhood: {result.listings[0].neighborhood}")

    except Exception as e:
        print(f"\n❌ An error occurred during generation: {e}")
        print("Please ensure your OPENAI_API_KEY is correct and you have API credits.")

if __name__ == "__main__":
    generate_listings()