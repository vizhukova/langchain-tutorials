import os
from config import openai_api_key
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

cwd = os.getcwd()  # Get the current working directory (cwd)

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
with open(cwd + '/data/PaulGrahamEssays/good.txt', 'r') as file:
    text = file.read()

# Printing the first 285 characters as a preview
print (text[:285])
# April 2008(This essay is derived from a talk at the 2008 Startup School.)About a month after we started Y Combinator we came up with the
# phrase that became our motto: Make something people want.  We've
# learned a lot since then, but if I were choosing now that's still
# the one I'd pick.

# how many tokens are in this document
num_tokens = llm.get_num_tokens(text)
print (f"There are {num_tokens} tokens in your file")
# There are 3970 tokens in your file

# if text is too big - it's need to be chunked into a smaller pices 
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
docs = text_splitter.create_documents([text])

print (f"You now have {len(docs)} docs intead of 1 piece of text")
# You now have 4 docs intead of 1 piece of text

# Get your chain ready to use
chain = load_summarize_chain(llm=llm, chain_type='map_reduce') # verbose=True optional to see what is getting sent to the LLM

# Use it. This will run through the 4 documents, summarize the chunks, then get a summary of the summary.
output = chain.run(docs)
print (output)

