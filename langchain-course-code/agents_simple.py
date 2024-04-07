import os
import openai
from langchain import hub
from langchain_openai.llms import OpenAI
from langchain.agents import  create_react_agent, load_tools, AgentExecutor

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

# llm_math = LLMMathChain.from_llm(llm=llm)
# math_tool = Tool(
#     name="Calculator",
#     func=llm_math.run,
#     description="Useful for when you need to answer questions related to Math."
# )
tools = load_tools(
        ['llm-math'],
        llm=llm
)

prompt = hub.pull("hwchase17/react")

print(tools[0].name, tools[0].description)

# TODO: change to new method
# ReAct framework = Reasoning and Action
agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = "If James is currently 45 years old, how old will he be in 50 years? \
    If he has 4 kids and adopted 7 more, how many children does he have?"
result = agent_executor.invoke({"input": query})
print(result['output'])

# print(f" ChatGPT ::: {llm.predict('what is 3.1^2.1')}")
