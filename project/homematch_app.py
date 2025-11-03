import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

LISTINGS_FILE = "listings.json"
PERSIST_DIRECTORY = "./chroma_db"

def load_environment():
    """Load .env, require API key, and optionally honor a custom base URL."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in .env file. Please create a .env file.")
        return False
    # Optional: custom base URL (e.g., Vocareum, Azure OpenAI-compatible endpoints)
    custom_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_HOST")
    if custom_base:
        os.environ["OPENAI_API_BASE"] = custom_base
        print(f"Using custom OPENAI_API_BASE: {custom_base}")
    return True

def load_and_prepare_listings():
    """
    Loads listings from JSON and converts them into LangChain Document objects
    for ingestion into the vector database.
    """
    print(f"Loading listings from {LISTINGS_FILE}...")
    try:
        with open(LISTINGS_FILE, "r", encoding="utf-8") as f:
            listings_data = json.load(f)["listings"]
    except FileNotFoundError:
        print(f"Error: '{LISTINGS_FILE}' not found.")
        print("Please run `python generate_listings.py` first or provide a listings.json file or use the offline version.")
        return None

    documents = []
    for i, listing in enumerate(listings_data):
        content = (
            f"Property Description: {listing['description']}\n"
            f"Neighborhood: {listing['neighborhood_description']}"
        )
        metadata = {
            "id": f"listing_{i}",
            "neighborhood": listing['neighborhood'],
            "price": listing['price'],
            "bedrooms": listing['bedrooms'],
            "bathrooms": listing['bathrooms'],
            "house_size_sqft": listing['house_size_sqft'],
            "full_description": listing['description'],
            "neighborhood_description": listing['neighborhood_description']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"Successfully loaded and prepared {len(documents)} documents.")
    return documents

def setup_vector_database(documents, embeddings, persist_directory=PERSIST_DIRECTORY, reset=True):
    """Initializes and persists the Chroma vector database."""
    if reset and os.path.isdir(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"✅ Vector database created and persisted at {persist_directory}.")
    return vectorstore

def get_buyer_preferences():
    """
    Returns a hard-coded set of buyer preferences as a single narrative string.
    Update this as needed to simulate different buyer profiles.
    """
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]
    profile = "Buyer Profile:\n" + "\n".join(f"- {a}" for a in answers)
    print("--- Buyer Preferences (Step 4) ---")
    print(profile)
    return profile

def create_personalization_chain(llm):
    """
    Creates a LangChain chain to generate personalized property descriptions
    based on buyer preferences and listing data.
    Update this template as needed to refine the personalization approach.
    """
    template = """
    You are an expert real estate copywriter at 'Future Homes Realty'.
    Your task is to rewrite a property listing to personally resonate with a specific buyer.

    CRITICAL RULE: Do NOT change any factual information (price, bedrooms, bathrooms, sqft, specific features).
    Only highlight how the existing features match the buyer's needs.

    Buyer Preferences:
    {buyer_preferences}

    Original Listing Data:
    - Neighborhood: {neighborhood}
    - Price: ${price:,}
    - Specs: {bedrooms} bed, {bathrooms} bath, {house_size_sqft} sqft
    - Original Description: {original_description}
    - Neighborhood Vibe: {neighborhood_description}

    Your Task:
    Write a new, personalized "HomeMatch" description for this buyer (approx 3-4 paragraphs).
    Start by directly addressing them (e.g., "Based on what you're looking for...") and show how this specific property fits.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

def main():
    print("🚀 Starting 'HomeMatch' Application...")
    if not load_environment():
        return

    chat_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    llm = ChatOpenAI(model=chat_model, temperature=0.5)
    embeddings = OpenAIEmbeddings(model=embed_model)

    documents = load_and_prepare_listings()
    if documents is None:
        return

    vectorstore = setup_vector_database(documents, embeddings, reset=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    buyer_profile = get_buyer_preferences()

    print("\n--- Performing Semantic Search (Step 5) ---")
    retrieved_docs = retriever.invoke(buyer_profile)
    print(f"Found {len(retrieved_docs)} matching listings.")

    print("\n--- Generating Personalized Descriptions (Step 6) ---")
    personalization_chain = create_personalization_chain(llm)

    for i, doc in enumerate(retrieved_docs):
        md = doc.metadata
        chain_input = {
            "buyer_preferences": buyer_profile,
            "neighborhood": md['neighborhood'],
            "price": md['price'],
            "bedrooms": md['bedrooms'],
            "bathrooms": md['bathrooms'],
            "house_size_sqft": md['house_size_sqft'],
            "original_description": md['full_description'],
            "neighborhood_description": md['neighborhood_description']
        }
        personalized_description = personalization_chain.invoke(chain_input)

        print(f"\n======= MATCH {i+1}: {md['neighborhood']} =======")
        print(f"Price: ${md['price']:,} | {md['bedrooms']} Bed | {md['bathrooms']} Bath")
        print("---")
        print("✨ Your Personalized 'HomeMatch' Description:")
        print(personalized_description)
        print("--------------------------------------------------")
        print("(For reference, Original Description:)")
        print(f"({md['full_description']})")
        print(f"==================================================")

if __name__ == "__main__":
    main()
