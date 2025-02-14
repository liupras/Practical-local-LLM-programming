#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-14
# @function: AI助理
# @version : V0.5
# @Description ：智能的调用多款工具解决实际问题。

# https://python.langchain.com/docs/how_to/agent_executor/

import os 
os.environ['USER_AGENT'] = 'agent_executor'

"""
1. 查天气的工具
"""
from langchain_core.tools import tool

@tool(parse_docstring=True)
def get_wheather_info(
    city_name: str = ''  #不设置默认值可能导致LLM强行解析city_name出错或者强行调用这个tool
) -> str:
    """获取某个城市的天气信息。如果没有可靠的依据来确定 city_name，则不要调用 get_wheather_info！

    Args:
        city_name: 城市名称。        
    """
    print(f'Getting weather information for:{city_name}')
    if not city_name:
        return "缺少 city_name 参数，无法检索天气信息。"        
        """
        **这个返回很重要**
        返回错误后，agent会放弃这个结果，用自己的能力回答问题，这样结果更加合理；
        否则，agent会使用空的city_name调用这个tool，并且会拼凑出new york的天气或者别的天气方面的信息。
        """
    else:
        return f"{city_name}的气温是25摄氏度。"


"""
确定重要文件路径
"""

import os,sys

# 将上级目录加入path，这样就可以引用上级目录的模块不会报错
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

def get_persist_directory(model_name):
    """矢量数据库存储路径"""

    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/es_{model_name}')

"""
在本地生成嵌入数据库
生成以后，反复测试时用起来比较方便。
"""

from common.MyVectorDB import LocalVectorDBChroma
def create_db(model_name):    
    """生成本地矢量数据库"""

    persist_directory = get_persist_directory(model_name)
    if os.path.exists(persist_directory):
        return

    db = LocalVectorDBChroma(model_name,persist_directory)    
    db.embed_webpage("http://wfcoding.com/articles/programmer/p0102/")

def test_search(embed_model_name,query):
    """查询矢量数据库"""

    persist_directory = get_persist_directory(embed_model_name)
    db = LocalVectorDBChroma(embed_model_name,persist_directory)
    vector_store = db.get_vector_store()

    results = vector_store.similarity_search_with_score(query,k=2)
    print(results)

"""
2. 创建一个检索器
"""

def create_retriever(embed_model_name):
    """创建检索器"""

    persist_directory = get_persist_directory(embed_model_name)
    db = LocalVectorDBChroma(embed_model_name,persist_directory)

    # 基于Chroma 的 vector store 生成 检索器
    vector_store = db.get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )
    return retriever

"""
3. 创建工具集 tools
"""

from langchain.tools.retriever import create_retriever_tool

def create_tools(embed_model_name):
    """创建工具集"""

    retriever_tool = create_retriever_tool(
        create_retriever(embed_model_name),
        "elastic_search",
        "只有当您搜索有关 elasticsearch 的知识时才能使用此工具！",
    )

    tools = [get_wheather_info, retriever_tool]
    return tools

"""
4. 创建智能体
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_ollama import ChatOllama

def create_agent(llm_model_name,embed_model_name):
    """创建智能体"""

    from langchain_core.tools import render_text_description

    tools = create_tools(embed_model_name)
    #rendered_tools = render_text_description(tools)
    #print(rendered_tools)
    
    # 此prompt是基于hwchase17/openai-functions-agent修改的
    systemprompt = """\
    您是一名助理，有权使用以下工具集。
    下面是每个工具的名称和说明：

    [get_wheather_info, elastic_search]

    - **仅在需要时使用上述工具集！**
    - 如果没有可靠的依据来确定 city_name，则不要调用 get_wheather_info！
    """ 
    prompt = ChatPromptTemplate([
        ("system", systemprompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

def create_agent_executor(llm_model_name,embed_model_name):
    """创建agent_executor"""

    tools = create_tools(embed_model_name)
    agent = create_agent(llm_model_name,embed_model_name)    

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    """实际上在create_agent中已经创建了tools，这里还要再传入tools，似乎有点多余。"""

    return agent_executor


from langchain.agents import AgentExecutor

def test_agent_executor(llm_model_name,embed_model_name,queries):
    """测试AgentExecutor"""

    print(f'--------{llm_model_name}----------')

    agent_executor = create_agent_executor(llm_model_name,embed_model_name)

    for query in queries:
        r = agent_executor.invoke({"input": query})
        print(f'agent_executor.invoke:\n{r}')

def test_agent_executor_stream(llm_model_name,embed_model_name,queries):
    """测试AgentExecutor"""

    print(f'--------{llm_model_name}----------')

    agent_executor = create_agent_executor(llm_model_name,embed_model_name)

    for query in queries:
        for chunk in agent_executor.stream({"input": query}):
            print(chunk)
            print("----")


"""
7. 记录会话
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def create_agent_executor_with_history(llm_model_name,embed_model_name):
    """创建记录聊天历史的智能体"""
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    agent_executor = create_agent_executor(llm_model_name,embed_model_name)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


# 将上级目录加入path，这样就可以引用上级目录的模块不会报错
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def create_agent_executor_with_history_2(llm_model_name,embed_model_name):
    """可以控制聊天历史长度的智能体"""
    
    # 处理聊天历史
    from common.LimitedChatMessageHistory import SessionHistory
    session_history = SessionHistory(max_size=10)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return session_history.process(session_id)
    
    agent_executor = create_agent_executor(llm_model_name,embed_model_name)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def test_agent_with_chat_history(llm_model_name,embed_model_name,session_id,queries):
    """测试记录会话功能"""

    print(f'--------{llm_model_name}----------')

    agent_with_chat_history = create_agent_executor_with_history(llm_model_name,embed_model_name)    

    conf = {"configurable": {"session_id": session_id}}
    for query in queries:
        r = agent_with_chat_history.invoke(
            {"input": query},
            config=conf,
        )
        print(f'agent_with_chat_history.invoked:\n{r}')
    

def test_agent_with_chat_history_stream(llm_model_name,embed_model_name,session_id,queries):
    """测试记录会话功能，流式输出"""

    print(f'--------{llm_model_name}----------')

    agent_with_chat_history = create_agent_executor_with_history(llm_model_name,embed_model_name)  

    conf = {"configurable": {"session_id": session_id}}
    for query in queries:
        for chunk in agent_with_chat_history.stream(
            {"input": query},
            config=conf,
        ):
            print(chunk)
            print("----")

if __name__ == '__main__':   
   
   queries = ["你好，你擅长能做什么？","上海的天气怎么样？","如何实现elasticsearch的深度分页？"]  
   test_agent_executor_stream("qwen2.5","shaw/dmeta-embedding-zh",queries)

   queries = ["您好，我叫刘大钧。","请问我叫什么名字？"]
   test_agent_with_chat_history("qwen2.5","shaw/dmeta-embedding-zh","liu123",queries)