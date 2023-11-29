
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

cwd = os.getcwd()  # Get the current working directory (cwd)

# Run this cell if you want to make your display wider
from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

loader = TextLoader(cwd + '/data/PaulGrahamEssays/disc.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

lm = ChatOpenAI()

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(lm, chain_type="map_reduce", verbose=True)
print(chain.run(texts))
