import os
from dotenv import load_dotenv
from langchain.document_loaders import HNLoader
from langchain.document_loaders import GutenbergLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
cwd = os.getcwd()  # Get the current working directory (cwd)

openai_api_key = os.getenv("OPENAI_API_KEY")

# https://github.com/openai/chatgpt-retrieval-plugin
# document loaders https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
# https://llamahub.ai/

# A class that extends the CheerioWebBaseLoader class. 
# It represents a loader for loading web pages from the Hacker News website.
loader = HNLoader("https://news.ycombinator.com/item?id=34422627")
data = loader.load()
print (f"Found {len(data)} comments")
print (f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:4]])}")
# Found 76 comments
# Here's a sample:

# Ozzie_osman 8 months ago  
#              | next [–] 

# LangChain is awesome. For people not sure what it's doing, large language models (LLMs) are very Ozzie_osman 8 months ago  
#              | parent | next [–] 

# Also, another library to check out is GPT Index (https://github.com/jerryjliu/gpt_index)



# Project Gutenberg is an online library of free eBooks.
# 
# This notebook covers how to load links to Gutenberg e-books into a document format that we can use downstream.
# 
loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/2148/pg2148.txt")
data = loader.load()
print(data[0].page_content[1855:1984])
print(data[0].metadata)
#  At Paris, just after dark one gusty evening in the autumn of 18-,
#  I was enjoying the twofold luxury of meditation 


# URLs and webpages
urls = [
    "http://www.paulgraham.com/",
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print('///////////////////')
print(data[0].page_content)
# 'New: \n\nHow to Do Great Work |\nRead |\nWill |\nTruth\n\n\n\n\n\nWant to start a startup? Get funded by Y Combinator.\n\n\n\n\n\n\n\n\n\n© mmxxiii pg'

# Text Splitters
# https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
# https://python.langchain.com/docs/modules/data_connection/document_transformers/

# This is a long document we can split up.
file_path = cwd + "/data/PaulGrahamEssays/worked.txt" 
with open(file_path) as f:
    pg_work = f.read()
    
print (f"You have {len([pg_work])} document")
# You have 1 document
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.create_documents([pg_work])
print (f"You have {len(texts)} documents")
# You have 610 documents
print ("Preview:")
print (texts[0].page_content, "\n")
print (texts[1].page_content)
# Preview:
# February 2021Before college the two main things I worked on, outside of school,
# were writing and programming. I didn't write essays. I wrote what 

# beginning writers were supposed to write then, and probably still
# are: short stories. My stories were awful. They had hardly any plot,