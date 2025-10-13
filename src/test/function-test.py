# Importing the library for OpenAI API
import openai
import os
import json
openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key 
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

def simple_function(input_string):
    return f"Function called with argument: {input_string}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "simple_function",
            "description": "A simple function that returns a string",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_string": {
                        "type": "string",
                        "description": "A string to pass to the function"
                    }
                },
                "required": ["input_string"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are an assistant that can call a simple function."},
    {"role": "user", "content": "Call the simple function with the argument 'Hello there, World!'."}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

tool_calls = response.choices[0].message.tool_calls

if tool_calls:
    available_functions = {
        "simple_function": simple_function,
    }

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        if function_name == 'simple_function':
            function_response = function_to_call(
                input_string=function_args.get("input_string"),
            )

        messages.append({
            "role": "assistant",
            "content": function_response,
        })

for message in messages:
    print(f"{message['role']}: {message['content']}")