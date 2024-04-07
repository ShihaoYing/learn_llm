import os
import openai
from langchain import hub
from langchain_openai.llms import OpenAI
from langchain.agents import Tool, load_tools, create_react_agent, AgentExecutor

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

# memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Second Generic Tool
prompt = PromptTemplate(
        input_variables=["query"],
        template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the LLM Tool
llm_tool = Tool(
        name="Language Model",
        func=llm_chain.run,
        description="Use this tool for general queries and logic"
)

tools = load_tools(
        ['llm-math'],
        llm=llm
)
tools.append(llm_tool)  # adding the new tool to our tools list

prompt_agent = hub.pull("hwchase17/react-chat")

# Conversational Agent
# conversational_agent = initialize_agent(
#         agent="conversational-react-description",
#         tools=tools,
#         llm=llm,
#         verbose=True,
#         max_iterations=3,
#         memory=memory
# )
conversational_agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt_agent
)

agent_executor = AgentExecutor(agent=conversational_agent, tools=tools, memory=memory, verbose=True)

query = "How old is a person born in 1917 in 2023"

query_two = "How old would that person be if their age is multiplied by 100?"

# print(conversational_agent.agent.llm_chain.prompt.template)

result = agent_executor.invoke({"input": query})
results = agent_executor.invoke({"input": query_two})
# print(result['output'])
