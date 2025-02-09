#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-11
# @function: ad_hoc tool
# @version : V0.5


# https://python.langchain.com/docs/how_to/tools_prompting/

from langchain_core.tools import tool

def create_tools():
    """创建tools"""
    @tool
    def add(x: int, y: int) -> int:
        """计算a和b的和。"""
        print (f"add is called...{x}+{y}")
        return x + y

    @tool
    def multiply(x: int, y: int) -> int:
        """计算a和b的乘积。"""
        print (f"multiply is called...{x}*{y}")
        return x * y
    
    tools = [add, multiply]

    for t in tools:
        print("--")
        print(t.name)
        print(t.description)
        print(t.args)

    return tools

tools = create_tools()

# 创建提示词
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description

rendered_tools = render_text_description(tools)
print(rendered_tools)

system_prompt = f"""\
您是一名助理，有权使用以下工具集。
以下是每个工具的名称和说明：

{rendered_tools}

根据用户输入，返回要使用的工具的名称和输入。
以 JSON blob 形式返回您的响应，其中包含“name”和“arguments”键。

“arguments”应该是一个字典，其中的键对应于参数名称，值对应于请求的值。
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

def too_call(model_name,query):
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)

    chain = prompt | llm
    message = chain.invoke({"input": query})
    print(f'response: \n{message.content}')  

from langchain_ollama import ChatOllama

# 将上级目录加入path，这样就可以直接引用上级目录的模块
import os,sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.MyJsonOutputParser import ThinkJsonOutputParser

def too_call_json(model_name,query):
    """以json格式输出"""
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)

    chain = prompt | llm | ThinkJsonOutputParser()
    message =chain.invoke({"input": query})
    print(f'JsonOutputParser: \n{message}')

# Invoking the tool
# The function will select the appropriate tool by name, and pass to it the arguments chosen by the mode
from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig

class ToolCallRequest(TypedDict):
    """invoke_tool 函数使用的参数格式。"""

    name: str
    arguments: Dict[str, Any]


def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """执行工具调用的函数。

    Args:
        tool_call_request: 包含键名和参数的字典。
            `name` 必须与已存在的工具名称匹配。
            `arguments` 是工具函数的参数。
        config: 这是 LangChain 使用的配置信息，其中包含回调、元数据等内容。

    Returns:
        requested tool 的输出
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

r = invoke_tool({"name": "multiply", "arguments": {"x": 3, "y": 5}})
print(f'test invoke_tool:{r}')

def invoke_chain(model_name,query):
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)

    chain = prompt | llm | ThinkJsonOutputParser() | invoke_tool
    result =chain.invoke({"input": query})
    print(f'invoke_chain:\n{result}')    

def invoke_chain_with_input(model_name,query):
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)

    from langchain_core.runnables import RunnablePassthrough

    chain = (
        prompt | llm | ThinkJsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)
    )
    result = chain.invoke({"input": query})
    print(f'invoke_chain with input:\n{result}')

if __name__ == '__main__':
    query = "3 * 12等于多少？"
    '''
    too_call("llama3.1",query)
    too_call("deepseek-r1",query)

    too_call_json("llama3.1",query)
    too_call_json("deepseek-r1",query)
    
    invoke_chain("llama3.1",query)
    invoke_chain("deepseek-r1",query)
    '''
    invoke_chain_with_input("llama3.1",query)
    invoke_chain_with_input("deepseek-r1",query)

    query = "11 + 49等于多少？"