#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-07
# @function: LLM调用tools计算加法和乘法
# @version : V0.5
# @Description ：调用函数tools计算加法和乘法。

# https://python.langchain.com/docs/how_to/tool_results_pass_to_model/

from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算a和b的和。"""
    print (f"add is called...{a}+{b}")
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算a和b的乘积。"""
    print (f"multiply is called...{a}*{b}")
    return a * b

tools = [add, multiply]


# 让模型调用一个工具，并把消息添加到历史记录中。
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

def caculate(model_name,query):
    print(f"\n---------{model_name}---------------")
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(query)]

    # 调用LLM，将query转化为json结构
    print("---1、调用LLM，将query转化为json结构---")
    ai_msg = llm_with_tools.invoke(messages)
    print(f' tool_calls is:{ai_msg.tool_calls}')

    messages.append(ai_msg)
    print(f'messages are:{messages}')

    # 使用模型生成的参数来调用工具函数

    # 调用 add 和 multiply，计算出结果
    print("---2、不调用LLM，直接调用 add 和 multiply，计算出结果---")
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    print(f'now,messages are:{messages}')

    # 调用LLM，生成流畅的答案
    print("---3、调用LLM，生成流畅的答案---")
    result = llm_with_tools.invoke(messages)
    print(f'result is:{result.content}')

if __name__ == '__main__':
    query = "3 * 12等于多少？ 11 + 49等于多少？"

    caculate("llama3.1",query)
    #caculate("deepseek-llm",query)