import os
import openai
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)
embeddings = OpenAIEmbeddings()

# 1. Load a pdf file
loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

# 2. Split the document into chunks
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
)
splits = text_splitter.split_documents(docs)
print(len(splits))
# =============== ==================== # 


# Real-world exampl with embeddings!
# Chroma db = #pip install chroma
from langchain_community.vectorstores import Chroma

persist_directory = "./data/db/chroma"

vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,  # openai embeddings
        persist_directory=persist_directory
)
# print(vectorstore._collection.count())

query = "what do they say about ReAct prompting method?"

docs_resp = vectorstore.similarity_search(query=query, k=3)

print(len(docs_resp))
print(docs_resp[0].page_content)

vectorstore.persist()  # save this for later usage!

# Embeddings - simpler example - compare similarity etc.
# text1 = "Kitty"
# text2 = "Rock"
# text3 = "Cat"

# embed1 = embeddings.embed_query(text1)
# embed2 = embeddings.embed_query(text2)
# embed3 = embeddings.embed_query(text3)
# # print(f"Embed1 == {embed1}")
# import numpy as np
# similarity = np.dot(embed1, embed3)
# print(f"Similarity %: {similarity*100}")
