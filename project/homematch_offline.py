#!/usr/bin/env python3
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ListingDoc:
    page_content: str
    metadata: dict

def load_listings(path: str) -> List[ListingDoc]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs: List[ListingDoc] = []
    for i, listing in enumerate(data["listings"]):
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
        docs.append(ListingDoc(page_content=content, metadata=metadata))
    return docs

def default_buyer_profile() -> str:
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]
    profile = "Buyer Profile:\n" + "\n".join(f"- {a}" for a in answers)
    return profile

def build_vector_index(docs: List[ListingDoc]):
    corpus = [d.page_content for d in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=1)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def retrieve_top_k(vectorizer, X, query: str, k: int) -> List[int]:
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X)[0]
    order = sims.argsort()[::-1]
    return order[:k].tolist()

FEATURE_KEYWORDS = {
    "schools": ["school", "schools"],
    "shopping": ["shopping", "market", "plaza", "grocery"],
    "backyard": ["backyard", "garden", "yard", "patio"],
    "garage": ["garage"],
    "energy": ["energy", "insulation", "solar", "hvac"],
    "transit": ["bus", "transit", "subway", "ferry"],
    "highway": ["highway", "express"],
    "bike": ["bike", "bikeshare", "cycle"],
    "restaurants": ["restaurant", "dining", "eateries"],
    "theaters": ["theater", "theatre", "arts", "galleries"],
    "parks": ["park", "trails", "greenbelt"],
}

def match_features(text: str) -> List[str]:
    hits = []
    lower = text.lower()
    for name, kws in FEATURE_KEYWORDS.items():
        if any(k in lower for k in kws):
            hits.append(name)
    return hits

def heuristic_personalization(buyer_profile: str, md: dict) -> str:
    listing_text = (md.get("full_description", "") + "\n" + md.get("neighborhood_description", "")).strip()
    matches = match_features(listing_text)
    bp_lines = [ln.strip(" -") for ln in buyer_profile.splitlines() if ln.strip().startswith("-")]

    para1 = (
        f"Based on what you're looking for, this {md['bedrooms']} bed, "
        f"{md['bathrooms']} bath, {md['house_size_sqft']} sqft home in {md['neighborhood']} "
        f"could be a strong match. Priced at ${md['price']:,}, it offers the essentials you highlighted, "
        f"without compromising on day-to-day comfort."
    )

    para2 = (
        "Inside, the existing features focus on livability—" 
        "a practical layout and a kitchen/living area that supports everyday routines. "
        "From your preferences, we paid close attention to details like a comfortable space for gathering, "
        "room to cook, and an overall calm atmosphere."
    )

    if matches:
        bullet_intro = "This listing also aligns with your interests in:"
        bullets = "\n".join(f"- {m}" for m in matches)
        para3 = f"{bullet_intro}\n{bullets}"
    else:
        para3 = (
            "The neighborhood context emphasizes convenience and relaxation, with amenities and connections "
            "that make daily life easier."
        )

    para4 = (
        f"Neighborhood notes: {md['neighborhood_description']}"
    )

    return "\n\n".join([para1, para2, para3, para4])


def main():
    ap = argparse.ArgumentParser(description="Offline HomeMatch: TF-IDF retrieval and heuristic personalization")
    ap.add_argument("--listings", default=None, help="Path to listings JSON. Defaults to listings.json or offline_listings.json")
    ap.add_argument("--k", type=int, default=3, help="Top-k listings to display")
    ap.add_argument("--interactive", action="store_true", help="Enter buyer preferences interactively")
    args = ap.parse_args()

    # Resolve listings path
    if args.listings:
        listings_path = args.listings
    elif os.path.exists("listings.json"):
        listings_path = "listings.json"
    elif os.path.exists("offline_listings.json"):
        listings_path = "offline_listings.json"
    else:
        raise SystemExit("No listings file found. Provide --listings or ensure listings.json/offline_listings.json exists.")

    docs = load_listings(listings_path)

    if args.interactive:
        print("Enter a short paragraph about your preferences (end with Ctrl-D / Ctrl-Z):")
        try:
            buyer_profile = "Buyer Profile:\n- " + input().strip()
        except EOFError:
            buyer_profile = default_buyer_profile()
    else:
        buyer_profile = default_buyer_profile()

    print("\n--- Buyer Preferences ---")
    print(buyer_profile)

    vectorizer, X = build_vector_index(docs)
    top_idx = retrieve_top_k(vectorizer, X, buyer_profile, args.k)

    print("\n--- Matches ---")
    for rank, idx in enumerate(top_idx, start=1):
        md = docs[idx].metadata
        desc = heuristic_personalization(buyer_profile, md)
        print(f"\n======= MATCH {rank}: {md['neighborhood']} =======")
        print(f"Price: ${md['price']:,} | {md['bedrooms']} Bed | {md['bathrooms']} Bath | {md['house_size_sqft']} sqft")
        print("---")
        print(desc)
        print("\n(Original Listing:)")
        print(md['full_description'])
        print("==================================================")

if __name__ == "__main__":
    main()
