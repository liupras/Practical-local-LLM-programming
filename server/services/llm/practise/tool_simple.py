#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-07
# @function: LLM调用tools计算加法和乘法
# @version : V0.5
# @Description ：调用函数tools计算加法和乘法。

# https://python.langchain.com/docs/how_to/tool_results_pass_to_model/

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    print (f"add is called...{a}+{b}")
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    print (f"multiply is called...{a}*{b}")
    return a * b

tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)

# Now, let's get the model to call a tool. We'll add it to a list of messages that we'll treat as conversation history
from langchain_core.messages import HumanMessage

query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(query)]

# 调用LLM，将query转化为json结构
print("---1、调用LLM，将query转化为json结构---")
ai_msg = llm_with_tools.invoke(messages)

print(f' tool_calls is:{ai_msg.tool_calls}')
messages.append(ai_msg)
print(f'messages are:{messages}')

# Next let's invoke the tool functions using the args the model populated!

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