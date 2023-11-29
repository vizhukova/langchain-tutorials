import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# 
# FUNCTION CALLING MODELS
# 
# Function calling models are similar to Chat Models. They are fine tuned to give structured data outputs.
# This comes in handy when you're making an API call to an external service or doing extraction.

chat = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=1, openai_api_key=openai_api_key)

output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="What’s the weather like in Boston right now?")
     ],
     functions=[{
         "name": "get_current_weather",
         "description": "Get the current weather in a given location",
         "parameters": {
             "type": "object",
             "properties": {
                 "location": {
                     "type": "string",
                     "description": "The city and state, e.g. San Francisco, CA"
                 },
                 "unit": {
                     "type": "string",
                     "enum": ["celsius", "fahrenheit"]
                 }
             },
             "required": ["location"]
         }
     }
     ]
)
print(output)
# AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_current_weather', 'arguments': '{\n  "location": "Boston, MA"\n}'}})
