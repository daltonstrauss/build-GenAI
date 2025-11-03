**HomeMatch: Personalized Real Estate Listings**

**Overview**
HomeMatch turns standard real estate listings into buyer‑personalized narratives. You can run it two ways:
- 📴 Offline mode (no API credits required): fast demo with TF‑IDF retrieval and heuristic personalization
- 🌐 Online mode (OpenAI + Chroma): full LLM generation, vector search, and personalized rewrite

**Project Structure**
- generate_listings.py — Generates listings with an LLM into listings.json (online)
- homematch_app.py — Loads listings, builds Chroma DB, retrieves top‑k, and personalizes with an LLM (online)
- homematch_offline.py — Fully offline: TF‑IDF retrieval + heuristic personalization (no APIs)
- offline_listings.json — Ready‑to‑use sample listings for offline runs
- requirements.txt — Online dependencies
- .env.example — Template for environment variables (API key and custom base URL)
- listings.json — Generated at runtime by generate_listings.py (online)
- chroma_db/ — Chroma persistence directory (generated)

**Prerequisites**
- Python 3.10+ recommended
- A virtual environment (venv)

**Set Up A Virtual Environment**
- macOS/Linux:
```
python -m venv venv
source venv/bin/activate
```
- Windows (PowerShell):
```
python -m venv venv
venv\Scripts\Activate.ps1
```

**Offline Mode (No API Key Needed) ✅**
- Install the single dependency:
```
pip install scikit-learn
```
- Run offline pipeline:
```
python homematch_offline.py
```
- Optional flags:
  - --k 5  (change top‑k results)
  - --listings path/to/your_listings.json  (defaults to listings.json if present; otherwise offline_listings.json)
  - --interactive  (enter your own buyer preferences via prompt)

What it does:
- Loads listings JSON
- Builds a TF‑IDF index and retrieves top‑k matches against the buyer profile
- Produces a clean, factual, personalized description with no external API calls

**Online Mode (OpenAI + Chroma) 🚀**
- Install dependencies:
```
pip install -r requirements.txt
```
- Configure your environment:
  1) Copy .env.example to .env
  2) Set your key and (optionally) custom base URL
```
OPENAI_API_KEY=sk-your-key
# Optional: a custom base URL (e.g., Vocareum or other OpenAI‑compatible gateways)
OPENAI_API_BASE=https://openai.vocareum.com/v1
# Optional: override models
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
- Step 1: Generate listings (creates listings.json)
```
python generate_listings.py
```
- Step 2: Run the app (builds vector DB, retrieves top‑3, personalizes)
```
python homematch_app.py
```
- Output: You’ll see the top‑3 matches with personalized descriptions printed in the console.

**Tip: No credits but want to test online flow?**
If you don’t have API credits to generate listings, you can still test the vector DB + personalization pipeline by copying the offline sample:
```
copy offline_listings.json listings.json   # Windows
cp offline_listings.json listings.json     # macOS/Linux
python homematch_app.py
```

**Troubleshooting**
- “API key not set” or immediate exit:
  - Ensure .env is created and OPENAI_API_KEY is present
  - If using a gateway, also set OPENAI_API_BASE
- LangChain prompt error in generate_listings.py (depending on versions):
  - If you see an error related to prompt partial variables, update prompt construction to use .partial(...). Or use the offline copy command above to proceed
- Chroma persistence issues:
  - The app resets the Chroma directory each run; if you need to preserve state, set reset=False in setup_vector_database (in code)
- Windows PowerShell execution policy prevents venv activation:
  - Run PowerShell as Administrator:  Set-ExecutionPolicy RemoteSigned

**Frequently Asked**
- Can I change top‑k matches online?
  - Yes. In code, change search_kwargs={"k": 3} in homematch_app.py
- Can I provide my own listings to the offline script?
  - Yes, pass --listings path/to/your.json as long as it follows the same schema
- Does the app support custom OpenAI base URLs?
  - Yes. Set OPENAI_API_BASE in .env; the app will use it automatically

**Credits**
Built to satisfy the HomeMatch brief, including synthetic data generation, vector database search, and buyer‑personalized rewrites, with an offline path for environments without API credits.
