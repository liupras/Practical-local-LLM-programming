#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-13
# @function: AI助理
# @version : V0.5
# @Description ：智能的调用多款工具解决实际问题。

# https://python.langchain.com/docs/how_to/agent_executor/

import os 
os.environ['USER_AGENT'] = 'agent_executor'

# Tool
from langchain_core.tools import tool

@tool(parse_docstring=True)
def get_wheather_info(
    city_name: str = ''  #不设置默认值可能导致LLM强行解析city_name出错或者强行调用这个tool
) -> str:
    """Get weather information for a city.

    Args:
        city_name: Name of the city.        
    """
    print(f'Getting weather information for:{city_name}')
    if not city_name:
        return "city_name parameter is missing. Cannot retrieve weather information."        
        """
        **这个返回很重要**
        返回错误后，agent会放弃这个结果，用自己的能力回答问题，这样结果更加合理；
        否则，agent会使用空的city_name调用这个tool，并且会拼凑出new york的天气或者别的天气方面的信息。
        """
    else:
        return f"{city_name}'s wheather is 15 degrees."

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0,verbose=True)

from langchain_core.messages import HumanMessage
def test_llm():
    response = llm.invoke([HumanMessage(content="hi!")])
    print(f'say hi:{response.content}')

def test_get_wheather_info():
    print(f'get_wheather_info schema:{get_wheather_info.get_input_schema().model_json_schema()}')
    print(f'get_wheather_info tool_call_schema:{get_wheather_info.tool_call_schema.model_json_schema()}')
    print(f'invoke get_wheather_info test:{get_wheather_info.invoke({"city_name": "Beijing"})}')

    tools = [
        get_wheather_info,
    ]
    llm_with_tools = llm.bind_tools(tools)

    query = 'what is the weather in Beijing?'
    ai_msg = llm_with_tools.invoke(query)
    print(f'get_wheather_info tool_calls:{ai_msg.tool_calls}')

# Retrieve

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings

def get_retriever():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    vector = Chroma.from_documents(documents, embedding_model)
    retriever = vector.as_retriever()
    return retriever

#print(f'retriever.invoke:{retriever.invoke("how to upload a dataset")}')

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    get_retriever(),
    "langsmith_search",
    "Use this tool only if you are searching for information about LangSmith!",
)

tools = [get_wheather_info, retriever_tool]
model_with_tools = llm.bind_tools(tools)

def test_tools():

    response = model_with_tools.invoke([HumanMessage(content='Hi!')])

    # 返回结果不理想：ContentString为空，却有ToolCalls
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")

    response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")


# Create the agent

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

def get_agent():
    
    # 此prompt是基于hwchase17/openai-functions-agent修改的
    systemprompt = """You are a helpful assistant with access to the following functions:

    [get_wheather_info,langsmith_search]. 

    - **Use them only if required**!
    - If there is no reliable basis for determining city_name, do not call get_wheather_info
    """
    prompt = ChatPromptTemplate([
        ("system", systemprompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=get_agent(), tools=tools)

def test_agent_executor():
    r = agent_executor.invoke({"input": "hi!"})
    print(f'agent_executor.invoke 1:\n{r}')

    r = agent_executor.invoke({"input": "what is the weather in sf?"})
    print(f'agent_executor.invoke 2:\n{r}')

    r = agent_executor.invoke({"input": "how to start LangSmith?"})
    print(f'agent_executor.invoke 3:\n{r}')


# Adding in memory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def test_agent_with_chat_history():
    session_id = "liupras"
    print("\n*****************************")
    r = agent_with_chat_history.invoke(
        {"input": "hi! I'm bob"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f'agent_with_chat_history.invoke 1:\n{r}')
    
    print("\n*****************************")
    r = agent_with_chat_history.invoke(
        {"input": "what's my name?"},
        config={"configurable": {"session_id": session_id}},
    )
    print(f'agent_with_chat_history.invoke 2:\n{r}')
    

def test_agent_with_chat_history_stream():
    session_id = "liupras"
    print("\n*****************************")
    for chunk in agent_with_chat_history.stream(
        {"input": "hi! I'm bob"},
        config={"configurable": {"session_id": session_id}},
    ):
        print(chunk)
        print("----")

    print("*****************************")
    for chunk in agent_with_chat_history.stream(
        {"input": "what's my name?"},
        config={"configurable": {"session_id": session_id}},
    ):
        print(chunk)
        print("----")

if __name__ == '__main__':
    test_agent_with_chat_history()
    #test_agent_with_chat_history_stream()