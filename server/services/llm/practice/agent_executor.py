#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-13
# @function: AI助理
# @version : V0.5
# @Description ：智能的调用多款工具解决实际问题。

# https://python.langchain.com/docs/how_to/agent_executor/

import os 
os.environ['USER_AGENT'] = 'myagent'

# Tool
from langchain_core.tools import tool

@tool(parse_docstring=True)
def get_wheather_info(
    city_name: str
) -> str:
    """Get weather information for a city.

    Args:
        city_name: Name of the city.        
    """
    print(f'Getting weather information for:{city_name}')
    return f"{city_name}'s wheather is 15 degrees."

# test
print(f'get_wheather_info schema:{get_wheather_info.get_input_schema().model_json_schema()}')
print(f'get_wheather_info tool_call_schema:{get_wheather_info.tool_call_schema.model_json_schema()}')
print(f'invoke get_wheather_info test:{get_wheather_info.invoke({"city_name": "Beijing"})}')

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0,verbose=True)

tools = [
    get_wheather_info,
]
llm_with_tools = llm.bind_tools(tools)

query = 'what is the weather in Beijing?'
# ai_msg = llm_with_tools.invoke(query)
#print(f'get_wheather_info tool_calls:{ai_msg.tool_calls}')

# Retrieve

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = Chroma.from_documents(documents, embedding_model)
retriever = vector.as_retriever()

#print(f'retriever.invoke:{retriever.invoke("how to upload a dataset")}')

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Use this tool only if you are searching for information about LangSmith!",
)


# LLM Test
from langchain_core.messages import HumanMessage

#response = llm.invoke([HumanMessage(content="hi!")])
#print(f'say hi:{response.content}')

# tools test
tools = [get_wheather_info, retriever_tool]
model_with_tools = llm.bind_tools(tools)

"""
response = model_with_tools.invoke([HumanMessage(content='Hi!)])

# 返回结果不理想：ContentString为空，却有ToolCalls
print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")


response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
"""

# Create the agent

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# hwchase17/openai-functions-agent 这个提示词在llama3上不准确：发送简单的问候词 hi 时，也会调用 get_wheather_info
"""
prompt = hub.pull("hwchase17/openai-functions-agent")
print(f'openai-functions-agent:{prompt}')
prompt_openai = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
print(f'prompt_openai-functions-agent:{prompt_openai}')
"""

# ily294/llama-structured-functions
#prompt_llama = hub.pull("ily294/llama-functioncalling-v3")
systemprompt = """You are a helpful assistant with access to the following functions. Use them if required:

[get_wheather_info,langsmith_search]

If one of the parameters for the function is empty, do not call this function and directly use the capabilities of the large language model itself to return the answer.
"""
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant with access to the following functions:[get_wheather_info,langsmith_search]. **Use them if required**!"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
r = agent_executor.invoke({"input": "hi!"})
print(f'agent_executor.invoke:{r}')

r = agent_executor.invoke({"input": "whats the weather in sf?"})
print(f'agent_executor.invoke:{r}')


