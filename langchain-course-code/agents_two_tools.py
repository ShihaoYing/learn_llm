import os
import openai
from langchain import hub
from langchain_openai.llms import OpenAI
from langchain.agents import Tool, load_tools, create_react_agent, AgentExecutor

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

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

prompt_agent = hub.pull("hwchase17/react")

# ReAct framework = Reasoning and Action
agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt_agent
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = "if I have 54 eggs and Mary has 10, and 5 more people have 12 eggs each.  \
    How many eggs to we have in total?"

result = agent_executor.invoke({"input": query})
print(result['output'])

print(agent.get_prompts())
