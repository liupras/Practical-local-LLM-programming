#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 如何将运行时值传递给工具
# @version : V0.5
# @Description ：如何将运行时值传递给工具。

# https://python.langchain.com/docs/how_to/tool_runtime/

from typing import List
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolArg, tool

user_to_pets = {}

@tool(parse_docstring=True)
def update_favorite_pets(
    pets: List[str], user_id: Annotated[str, InjectedToolArg]
) -> None:
    """添加或者更新最喜爱的宠物列表。

    Args:
        pets: 最喜爱的宠物列表。
        user_id: 用户ID。
    """
    print(f'update_favorite_pets is called:{user_id}')
    user_to_pets[user_id] = pets


@tool(parse_docstring=True)
def delete_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """删除喜爱的宠物列表。

    Args:
        user_id: 用户 ID。
    """
    print(f'delete_favorite_pets is called:{user_id}')
    if user_id in user_to_pets:
        del user_to_pets[user_id]


@tool(parse_docstring=True)
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """列出最喜欢的宠物。

    Args:
        user_id: 用户 ID。
    """
    print(f'list_favorite_pets is called:{user_id}')
    return user_to_pets.get(user_id, [])

def test_tool():
    """测试工具"""

    # 查看这些工具的输入数据结构，我们会看到 user_id 仍然会列出来
    print(f'get_input_schema:{update_favorite_pets.get_input_schema().model_json_schema()}')

    # 但是如果我们查看工具调用数据结构（即传递给模型进行工具调用的内容），user_id 已被删除
    print(f'tool_call_schema:{update_favorite_pets.tool_call_schema.model_json_schema()}')

    user_id = "123"
    update_favorite_pets.invoke({"pets": ["lizard", "dog"], "user_id": user_id})
    print(f'user_to_pets:{user_to_pets}')
    print(f'list_favorite_pets.invoke:{list_favorite_pets.invoke({"user_id": user_id})}')

# 当模型调用该工具时，不会生成任何 user_id 参数/实参
tools = [
    update_favorite_pets,
    delete_favorite_pets,
    list_favorite_pets,
]

from langchain_ollama import ChatOllama

def invoke_tool(model_name,query):
    """测试生成的tool_call"""

    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    ai_msg = llm_with_tools.invoke(query)
    print(f'result:\n{ai_msg.tool_calls}')

    return ai_msg

# 在运行时注入参数
from copy import deepcopy

from langchain_core.runnables import chain

user_id ="u123"

@chain
def inject_user_id(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["user_id"] = user_id
        tool_calls.append(tool_call_copy)
    return tool_calls

def test_inject_user_id(model_name,query):
    ai_msg = invoke_tool(model_name,query)
    new_args = inject_user_id.invoke(ai_msg)
    print(f'inject_user_id:\n{new_args}')


tool_map = {tool.name: tool for tool in tools}

@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]

def execute_tool(model_name,query):
    """调用工具，返回结果"""

    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    # 将模型、注入用户ID代码和实际的工具链接在一起，创建工具执行链
    chain = llm_with_tools | inject_user_id | tool_router.map()

    result = chain.invoke(query)
    print(f'chain.invoke:\n{result}')
    print(f'now user_to_pets :\n{user_to_pets}')

if __name__ == '__main__':

    test_tool()

    query = "刘大军最喜欢的动物是狗和蜥蜴。"
    invoke_tool('llama3.1',query)
    invoke_tool('MFDoom/deepseek-r1-tool-calling:7b',query)

    test_inject_user_id('llama3.1',query)
    test_inject_user_id('MFDoom/deepseek-r1-tool-calling:7b',query)

    execute_tool('llama3.1',query)
    execute_tool('MFDoom/deepseek-r1-tool-calling:7b',query)
    