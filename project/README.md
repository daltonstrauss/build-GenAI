# HomeMatch: Personalized Real Estate Listings

Welcome to "HomeMatch," a proof-of-concept application by Future Homes Realty. This project uses Large Language Models (LLMs) and a vector database to transform standard real estate listings into personalized narratives that resonate with a buyer's unique preferences.

## Project Structure

-   `README.md`: This file.
-   `requirements.txt`: A list of all necessary Python packages.
-   `.env`: A file you must create to store your OpenAI API key (e.g., `OPENAI_API_KEY=sk-YourKeyHere`).
-   `listings.json`: This file will be created by the `generate_listings.py` script. It contains the 10+ synthetic property listings.
-   `generate_listings.py`: A script to generate the synthetic real estate listings using an LLM and save them to `listings.json`.
-   `homematch_app.py`: The main application. It loads the listings, stores them in a ChromaDB vector database, collects buyer preferences, performs a semantic search, and generates personalized descriptions.
-   `/chroma_db`: A directory that will be created by `homematch_app.py` to persist the vector database.

## 🚀 How to Run

1.  **Set up the Environment:**
    * Create a Python virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    * Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Set Your API Key:**
    * Create a file named `.env` in the root of the project.
    * Add your OpenAI API key to it:
        ```
        OPENAI_API_KEY=sk-YourKeyHere
        ```

3.  **Step 1: Generate Listings:**
    * Run the listing generation script. This will use the LLM to create 10 realistic listings and save them to `listings.json`.
        ```bash
        python generate_listings.py
        ```

4.  **Step 2: Run the HomeMatch Application:**
    * Run the main application. This will load the listings, build the database, and run the personalization pipeline based on the hard-coded buyer preferences.
        ```bash
        python homematch_app.py
        ```
    * The console will output the 3 most relevant listings, each with a new, personalized description.

## 💡 Stand-Out Suggestion: Multimodal Search

This project fully meets the core requirements. To implement the "Stand-Out Suggestion" (multimodal search with CLIP):

1.  **Modify Data Generation:** The `Listing` Pydantic model in `generate_listings.py` would be updated to include an `image_prompt` field (e.g., "A photo of a modern kitchen with a large granite island").
2.  **Embed Images:** Use a multimodal embedding model (like `CLIPModel` from `transformers`) to generate embeddings for these image prompts (or actual images, if you had them).
3.  **Multimodal Database:** Store both the `text_embedding` (from OpenAI) and the `image_embedding` (from CLIP) in ChromaDB.
4.  **Search Logic:** When a buyer searches, embed their preference text using *both* the OpenAI text model *and* the CLIP text model.
5.  **Hybrid Search:** Query the database using both embeddings (text-vs-text and text-vs-image) and use a re-ranking algorithm (like Reciprocal Rank Fusion) to combine the results, surfacing listings that are both textually and visually relevant.