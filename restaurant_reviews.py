import openai
import os

# Prompting the user to enter the restaurant name and cuisine type
restaurant_name = input("Enter the restaurant name: ")
cuisine_type = input("Enter the cuisine type: ")

# Create the prompt template based on the user inputs
prompt_template = f"Provide a summary of customer sentiments for {restaurant_name}, focusing on their {cuisine_type} dishes. Highlight key sentiments and mention any standout dishes or services."
print(f"Generated prompt: {prompt_template}")

# Configure OpenAI API
openai.api_base = "https://openai.vocareum.com/v1"

# Define your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

def generate_restaurant_review(prompt_template):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a restaurant critic. You are writing about reviews of restaurants. "
                },
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # The response is a JSON object containing more information than the generated review. We want to return only the message content
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Generating the response from the model
review_summary = generate_restaurant_review(prompt_template)

# Printing the output
print("Generated review:")
print(review_summary)