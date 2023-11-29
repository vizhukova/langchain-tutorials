import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# 
# PROMPTS
# Prompts - Text generally used as instructions to your model
# 

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

# I like to use three double quotation marks for my prompts because it's easier to read
prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""
print(llm(prompt))
# The statement is incorrect. Tomorrow is Tuesday, not Wednesday.

# 
# Prompt Template
# An object that helps create prompts based on a combination of user input, other non-static information and a fixed template string.
# 
llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

# Notice "location" below, that is a placeholder for another value later
template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location='Rome')

print (f"Final Prompt: {final_prompt}")
print ("-----------")
print (f"LLM Output: {llm(final_prompt)}")
# Final Prompt: 
# I really want to travel to Rome. What should I do there?
# Respond in one short sentence
# -----------
# LLM Output: Visit the Colosseum, the Vatican, and the Trevi Fountain.

# 
# Example Selectors
# An easy way to select from a series of examples that allow you to dynamic place in-context information into your prompt.
# Often used when your task is nuanced or you have a large list of examples.
#  https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "hand"},
    {"input": "pilot", "output": "helm"},
    {"input": "driver", "output": "pedal"},
    {"input": "tree", "output": "chair"},
    {"input": "bird", "output": "tree"},
]

# SemanticSimilarityExampleSelector will select examples that are similar to your input by semantic meaning

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples, 
    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key=openai_api_key), 
    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma, 
    
    # This is the number of examples to produce.
    k=2
)

similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    
    # Your prompt
    example_prompt=example_prompt,
    
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["noun"],
)

# Select a noun!
my_noun = "eye"
# my_noun = "student"

print(similar_prompt.format(noun=my_noun))
print(llm(similar_prompt.format(noun=my_noun)))