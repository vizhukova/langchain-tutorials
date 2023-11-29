import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_chain
from typing import Sequence
import enum

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Fuctions
# This is recommended method when starting out.
# specify a Pydantic schema and get a structured output
# function calling models doc https://platform.openai.com/docs/guides/gpt/function-calling
# 

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = create_structured_output_chain(Person, llm, prompt)
print(chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
))
# name='Sally' age=13 fav_food='Not specified'



# 
# Make it identify all people
# 
class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person] = Field(..., description="The people in the text")
chain = create_structured_output_chain(People, llm, prompt)
print(chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
))
# people=[Person(name='Sally', age=13, fav_food='unknown'), Person(name='Joey', age=12, fav_food='spinach'), Person(name='Caroline', age=23, fav_food='unknown')]

# 
# ENUM
# 


class Product(str, enum.Enum):
    CRM = "CRM"
    VIDEO_EDITING = "VIDEO_EDITING"
    HARDWARE = "HARDWARE"

class Products(BaseModel):
    """Identifying products that were mentioned in a text"""
    products: Sequence[Product] = Field(..., description="The products mentioned in a text")

chain = create_structured_output_chain(Products, llm, prompt)
print(chain.run(
    "The CRM in this demo is great. Love the hardware. The microphone is also cool. Love the video editing"
))

