import os
from dotenv import load_dotenv
from langchain.schema import Document
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# 
# DOCUMENT TYPE
# 
# An object that holds a piece of text and metadata (more information about that text)
# 
document = Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })

print(document)
# Document(page_content="This is my document. It is full of text that I've gathered from other places")
