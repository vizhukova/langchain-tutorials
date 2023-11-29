import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")

