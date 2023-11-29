# Some applications will require not just a predetermined chain of calls to LLMs/other tools, 
# but potentially an unknown chain that depends on the user's input. 
# In these types of chains, there is a “agent” which has access to a suite of tools. 
# Depending on the user input, the agent can then decide which, if any, of these tools to call.

# https://python.langchain.com/docs/modules/agents/

import os
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERP_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
response = agent({"input":"what was the first album of the" 
                    "band that Natalie Bergman is a part of?"})