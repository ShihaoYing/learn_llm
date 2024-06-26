import os
# from dotenv import find_dotenvotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.9, model=llm_model)
open_ai = OpenAI(temperature=0.7)


# LLMChain
prompt = PromptTemplate(
    input_variables=["language"],
    template="How do you say good morning in {language}"
)

chain = LLMChain(llm=open_ai, prompt=prompt)
print(chain.invoke(input=["German"])["text"])
