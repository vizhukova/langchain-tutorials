import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader('data/PaulGrahamEssays/worked.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

print (f"You have {len(texts)} documents")
# You have 78 documents

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)

#  Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()

print(retriever)
# VectorStoreRetriever(tags=['FAISS'], vectorstore=<langchain.vectorstores.faiss.FAISS object at 0x7f8389169070>)

docs = retriever.get_relevant_documents("what types of things did the author want to build?")

print("\n\n".join([x.page_content[:200] for x in docs[:2]]))
# standards; what was the point? No one else wanted one either, so
# off they went. That was what happened to systems work.I wanted not just to build things, but to build things that would
# last.In this di

# much of it in grad school.Computer Science is an uneasy alliance between two halves, theory
# and systems. The theory people prove things, and the systems people
# build things. I wanted to build things. 

# 
# VECTOR STORES
# 
# Databases to store vectors. Most popular ones are Pinecone & Weaviate. 
# https://github.com/openai/chatgpt-retrieval-plugin#choosing-a-vector-database

# Conceptually, think of them as tables w/ a column for embeddings (vectors) and a column for metadata.

embedding_list = embeddings.embed_documents([text.page_content for text in texts])
print (f"You have {len(embedding_list)} embeddings")
print (f"Here's a sample of one: {embedding_list[0][:3]}...")
# You have 78 embeddings
# Here's a sample of one: [-0.001058628615053026, -0.01118234211553424, -0.012874804746266883]...