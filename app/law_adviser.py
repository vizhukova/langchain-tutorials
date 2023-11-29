import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
cwd = os.getcwd()  # Get the current working directory (cwd)

openai_api_key = os.getenv("OPENAI_API_KEY")
urls = ["https://zakon.rada.gov.ua/laws/show/254%D0%BA/96-%D0%B2%D1%80"]

try:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    if not data:
        print("No data returned.")
except Exception as e:
    print(f"An error occurred: {e}")


print (f"You have {len(data)} document")
print (f"You have {len(data[0].page_content)} characters in that document")

# split our long doc into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(data)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])
print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)

# Create your retrieval engine
llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key,  max_tokens=200)

template = '''
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
{context}
Question: {question}
Answer:
'''
prompt = PromptTemplate(
    template=template, 
    input_variables=[
        'context',
        'question'
    ]
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt
    })

query = "Мені не дають відпустку"

result = qa.run(query)
print(result)