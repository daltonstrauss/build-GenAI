# Importing the necessary library for OpenAI API
import openai
import os
openai.api_base = "https://openai.vocareum.com/v1"

# Define your OpenAI API key 
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

product_name = "EcoLight Smart Bulb"
cool_feature = "Energy-saving with customizable colors and voice control"
audience_persona = "Environmentally conscious homeowners"

prompt = f"Create a catchy, clever 140 character social media post targeted toward {audience_persona} introducing and promoting a new {cool_feature} feature of {product_name}."
print(prompt)

def generate_social_media_post(prompt):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        # Use openai.ChatCompletion.create for openai < 1.0
        # openai.chat.completions.create for openai > 1.0
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
          {
            "role": "system",
            "content": "You are a social media influencer and writer. "
          },
          {
            "role": "user",
            "content": prompt
          }
          ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        # The response is a JSON object containing more information than the generated post. We want to return only the message content
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Generating the social media post
generated_post = generate_social_media_post(prompt)

# Printing the output. 
print("Generated Social Media Post:")
print(generated_post)