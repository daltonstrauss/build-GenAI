# Restaurant Review Sentiment Generator
This Python script uses OpenAI's GPT-3.5 model to generate a summary of customer sentiments for a specific restaurant and its cuisine type. Simply provide the restaurant's name and cuisine type, and the script will create a customized restaurant review based on the input.
## Features
- Dynamically prompts the user for the restaurant name and cuisine type.
- Automatically generates a review summary focused on the provided cuisine type.
- Highlights key sentiments and mentions standout dishes or services.
- Utilizes OpenAI's gpt-3.5-turbo model for high-quality and concise review generation.
---
## Requirements
### Tools and Libraries
- Python 3.7 or higher
- OpenAI Python SDK (openai)
- A valid OpenAI API key
### Installation

Before running the script, install the required dependencies. You can do this with the following command:
```bash
pip install openai
```
---
## Setting Up
1. Get an OpenAI API Key 
   You’ll need an API key to use OpenAI's API. Visit OpenAI and sign up to get your key.
2. Set the API Key in Your Environment 
   Add your API key as an environment variable:
   
```bash
export OPENAI_API_KEY=your_openai_api_key
```

   * Replace your_openai_api_key with your actual API key.
   Alternatively, you can directly assign your API key to the api_key variable in the script, but keeping it as an environment variable is more secure.

3. Set the OpenAI Base URL 
   The script uses a custom OpenAI API base URL: https://openai.vocareum.com/v1. Ensure this URL is functional, and modify as necessary based on your OpenAI setup.
---
## How to Use
1. Run the Script 
   Execute the script using Python:
   
```bash
python restaurant_review_generator.py
```   
2. Input Details 
   - When prompted, enter the name of the restaurant.
   - Next, enter the cuisine type associated with the restaurant (e.g., "Italian", "Japanese", "New American", etc.).
3. View the Generated Review 
   The script will call OpenAI's API to generate a review based on your inputs. The output will be displayed in your terminal. You’ll see something like this:
   
```plaintext
Enter the restaurant name: Alinea
Enter the cuisine type: new american
Generated review:
Alinea is widely regarded as a top destination for New American cuisine. Many customers rave about their innovative and artistic approach to dishes. Standout items include their edible balloons and their signature lamb dish. The service is attentive and enhances the overall dining experience. Guests frequently mention the immersive nature of meals and the creativity in menu presentation. While on the pricier side, it is described as well worth the experience.
```   
---
## Script Details
The script includes the following functionality:
- Dynamic Input: Prompts the user to provide the restaurant name and the type of cuisine they want to generate a review for.
- API Integration: Communicates with OpenAI's GPT-3.5 API to process your inputs and generate a detailed review.
- Error Handling: Returns a clear error message if there are issues while communicating with the API.
---
## Example
### Input:
```plaintext
Enter the restaurant name: Sushi Nakazawa
Enter the cuisine type: Japanese
```
### Output:
```plaintext
Generated review:
Sushi Nakazawa is often praised as one of the top sushi establishments. Guests frequently comment on the incredible precision and artistry in their omakase offerings. The chefs expertly handle each dish, with standout items such as the uni sushi and toro sashimi. Service is top-notch, with the staff being attentive and knowledgeable. While reservations can be challenging to secure, the overall dining experience is truly worth the effort for Japanese cuisine enthusiasts.
```
---
## Notes
- This script is designed for educational and illustrative purposes. Modify it according to your needs.
- Ensure that the OpenAI subscription plan supports API calls to the gpt-3.5-turbo model, as running the script repeatedly may incur costs.
---
## Troubleshooting
### Common Errors:
1. openai.error.AuthenticationError:
   - Ensure that your OpenAI API key is set correctly.
   - Double-check the environment variable or API key declaration in the script.
2. Empty or Incorrect Responses:
   - Verify that the OpenAI base URL (`https://openai.vocareum.com/v1`) is correct.
   - Ensure that the model parameter (gpt-3.5-turbo) is available under your OpenAI subscription.
3. Other Exceptions:
   - If an undefined error occurs, review the specific error message printed by the script for further debugging.
---
## License
The project is open-source and free to use under the MIT License.
---
Enjoy your restaurant sentiment analysis! :blush:
---