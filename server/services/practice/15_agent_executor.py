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

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def test_llm(model_name,query):
    """测试大模型能否聊天"""

    llm = ChatOllama(model=model_name,temperature=0,verbose=True)
    response = llm.invoke([HumanMessage(content=query)])
    print(f'{model_name} answer:\n{response.content}')

def test_get_wheather_info(llm_model_name,city_name):
    """测试获取天气信息"""

    print(f'--------{llm_model_name}----------')

    """
    print(f'get_wheather_info schema:{get_wheather_info.get_input_schema().model_json_schema()}')
    print(f'get_wheather_info tool_call_schema:{get_wheather_info.tool_call_schema.model_json_schema()}')
    print(f'invoke get_wheather_info test:{get_wheather_info.invoke({"city_name": city_name})}')
    """

    tools = [
        get_wheather_info,
    ]

    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    query = f'{city_name}的天气怎么样？'
    ai_msg = llm_with_tools.invoke(query)
    print(f'get_wheather_info tool_calls:\n{ai_msg.tool_calls}')


"""
2. 确定重要文件路径
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
3.在本地生成嵌入数据库
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
4. 创建一个检索器
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
5. 创建工具集 tools
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


def test_tools(llm_model_name,embed_model_name,queries):
    """测试工具集"""
    
    llm = ChatOllama(model=llm_model_name,temperature=0,verbose=True)
    tools = create_tools(embed_model_name)
    llm_with_tools = llm.bind_tools(tools)

    print(f'--------{llm_model_name}----------')

    for query in queries:
        response = llm_with_tools.invoke([HumanMessage(content=query)])
        print(f"ContentString:\n {response.content}")
        print(f"ToolCalls: \n{response.tool_calls}")


"""
6. 创建智能体
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent

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

from langchain.agents import AgentExecutor

def create_agent_executor(llm_model_name,embed_model_name):
    """创建agent_executor"""

    tools = create_tools(embed_model_name)
    agent = create_agent(llm_model_name,embed_model_name)    

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    """实际上在create_agent中已经创建了tools，这里还要再传入tools，似乎有点多余。"""

    return agent_executor

def test_agent_executor(llm_model_name,embed_model_name,queries):
    """测试AgentExecutor"""

    print(f'--------{llm_model_name}----------')

    agent_executor = create_agent_executor(llm_model_name,embed_model_name)

    for query in queries:
        r = agent_executor.invoke({"input": query})
        print(f'agent_executor.invoke:\n{r}')


def test(llm_model_name,embed_model_name):
    """集中测试方法"""

    create_db(embed_model_name)

    query = "如何实现elasticsearch的深度分页？"
    test_search(embed_model_name,query)

    query = "你好，你擅长能做什么？"

    test_llm(llm_model_name,query)
    test_get_wheather_info(llm_model_name,"深圳")

    queries = ["你好，你擅长能做什么？","上海的天气怎么样？","如何实现elasticsearch的深度分页？"]

    test_tools(llm_model_name,embed_model_name,queries)    
    test_agent_executor(llm_model_name,embed_model_name,queries)

if __name__ == '__main__':   

   
   test_get_wheather_info("qwen2.5","北京") 
   test_get_wheather_info("llama3.1","北京")
   test_get_wheather_info("MFDoom/deepseek-r1-tool-calling:7b","北京")


   queries = ["你好，你擅长能做什么？","上海的天气怎么样？","如何实现elasticsearch的深度分页？"]
   test_tools("qwen2.5","shaw/dmeta-embedding-zh",queries)
   test_tools("llama3.1","shaw/dmeta-embedding-zh",queries)
   test_tools("MFDoom/deepseek-r1-tool-calling:7b","shaw/dmeta-embedding-zh",queries)

   test_agent_executor("qwen2.5","shaw/dmeta-embedding-zh",queries)
   test_agent_executor("llama3.1","shaw/dmeta-embedding-zh",queries)
   test_agent_executor("MFDoom/deepseek-r1-tool-calling:7b","shaw/dmeta-embedding-zh",queries)
   
   test("qwen2.5","shaw/dmeta-embedding-zh")
   test("llama3.1","shaw/dmeta-embedding-zh")
   test("MFDoom/deepseek-r1-tool-calling:7b","shaw/dmeta-embedding-zh")