Rubric
Use this project rubric to understand and assess the project criteria.

Synthetic Data Generation
Criteria	Submission Requirements
Generating Real Estate Listings with an LLM

The submission must demonstrate using a Large Language Model (LLM) to generate at least 10 diverse and realistic real estate listings containing facts about the real estate.

Semantic Search
Criteria	Submission Requirements
Creating a Vector Database and Storing Listings

The project must demonstrate the creation of a vector database and successfully storing real estate listing embeddings within it. The database should effectively store and organize the embeddings generated from the LLM-created listings.

Semantic Search of Listings Based on Buyer Preferences

The application must include a functionality where listings are semantically searched based on given buyer preferences. The search should return listings that closely match the input preferences.

Augmented Response Generation
Criteria	Submission Requirements
Logic for Searching and Augmenting Listing Descriptions

The project must demonstrate a logical flow where buyer preferences are used to search and then augment the description of real estate listings. The augmentation should personalize the listing without changing factual information.

Use of LLM for Generating Personalized Descriptions

The submission must utilize an LLM to generate personalized descriptions for the real estate listings based on buyer preferences. The descriptions should be unique, appealing, and tailored to the preferences provided.

Suggestions to Make Your Project Stand Out
For a project that truly stands out, consider integrating CLIP to enable multimodal search capabilities. This advanced feature would allow the application to search real estate listings through textual descriptions and images associated with each property. By doing so, the application can align visual elements of a property (like style, layout, and surroundings) with the textual buyer preferences.

Implementation Overview

Image Embeddings: Generate embeddings for real estate images using CLIP, which can then be stored in the vector database alongside text embeddings.

Multimodal Search Logic: Develop a search algorithm that considers both text and image embeddings to find listings that best match the buyer's preferences, including visual aspects.