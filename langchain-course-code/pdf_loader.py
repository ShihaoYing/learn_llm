import os
import openai
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI

from langchain_community.document_loaders import PyPDFLoader

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)

loader = PyPDFLoader("./data/react-paper.pdf")
pages = loader.load()

# print(len(pages))

page = pages[0]
# print(pages)
print(page.page_content)  # first 700 characters on the page
print(page.metadata)
