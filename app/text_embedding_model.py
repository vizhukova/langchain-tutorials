import os
from config import openai_api_key
from langchain import OpenAI
# The vectorstore we'll be using
from langchain.vectorstores import FAISS
# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
# The easy document loader for text
from langchain.document_loaders import TextLoader
# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

cwd = os.getcwd()  # Get the current working directory (cwd)

# 
# TEXT EMBEDDING MODEL
# Change your text into a vector (a series of numbers that hold the semantic 'meaning' of your text). 
# Mainly used when comparing two pieces of text together.

# Semantic means 'relating to meaning in language or logic.'

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
text = "Hi! It's time for the beach"
text_embedding = embeddings.embed_query(text)
print (f"Here's a sample: {text_embedding[:5]}...")
print (f"Your embedding is length {len(text_embedding)}")
# Here's a sample: [-0.00019600906371495047, -0.0031846734422911363, -0.0007734206914647714, -0.019472001962491232, -0.015092319017854244]...
# Your embedding is length 1536

#  It's the process of splitting your text, embedding the chunks, putting the embeddings in a DB, and then querying them.

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
loader = TextLoader(cwd + '/data/PaulGrahamEssays/worked.txt')
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document")
# You have 1 document
# You have 74663 characters in that document

# split our long doc into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])
print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")
# Now you have 29 documents that have an average of 2,930 characters (smaller pieces)

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)

# Create your retrieval engine

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
)
# Now it's time to ask a question. The retriever will go get the similar documents and combine with your question for the LLM to reason through.

# Note: It may not seem like much, but the magic here is that we didn't have to pass in our full original document.

query = "What does the author describe as good work?"
result = qa.run(query)
print(result)
# ' The author describes painting as good work.'