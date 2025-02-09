#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 在Agent中增加人的干预
# @version : V0.5
# @Description ：在Agent中增加人的干预。有些工具我们不信任模型能够自行执行。在这种情况下，我们可以做的一件事是在调用工具之前要求人工批准。

# https://python.langchain.com/docs/how_to/tools_human/

from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

def create_tools():
    @tool
    def count_emails(last_n_days: int) -> int:
        """计算电子邮件数量的函数。"""
        print(f'count_emails is called:{last_n_days}')
        return last_n_days * 2

    @tool
    def send_email(message: str, recipient: str) -> str:
        """发送电子邮件的函数。"""
        print(f'send_email is called:{recipient}:{message}')
        return f"邮件已经成功发送至：{recipient}."


    tools = [count_emails, send_email]

    return tools

tools = create_tools()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def test_tool_call(model_name,query):
    """测试tool_call，看看输出内容"""
    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    print(f' tool_calls is:\n{ai_msg.tool_calls}')

def call_tools(msg: AIMessage) -> List[Dict]:
    """调用工具方法。"""

    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls

def tool_call(model_name,query):
    """使用chain调用tools很方便。这里直接输出json格式的结果"""

    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)
    chain = llm_with_tools | call_tools
    result = chain.invoke(query)
    print(f'chain.invoked:\n{result}')


import json

class NotApproved(Exception):
    """自定义异常。"""
    print(f'Not approved:{Exception}')


def human_approval(msg: AIMessage) -> AIMessage:
    """负责传递其输入或引发异常。

    Args:
        msg: 聊天模型的输出

    Returns:
        msg: 消息的原始输出
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"您是否同意以下工具调用\n\n{tool_strs}\n\n"
        "除'Y/Yes'（不区分大小写）之外的任何内容都将被视为否。\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        print("主人没有批准。")
        raise NotApproved(f"未批准使用工具:\n\n{tool_strs}")
    print("主人已批准。")
    return msg

def approval(model_name,query):
    """由人类批准是否使用工具"""

    llm = ChatOllama(model=model_name,temperature=0.1,verbose=True)
    llm_with_tools = llm.bind_tools(tools)

    chain = llm_with_tools | human_approval | call_tools

    try:
        result = chain.invoke(query)
        print(f'human-in-the-loop chain.invoke:{result}')
    except NotApproved as e:
        print(f'Not approved:{e}')

if __name__ == '__main__':
    query = "我过去7天收到了多少封电子邮件？"

    test_tool_call("llama3.1",query)
    test_tool_call("MFDoom/deepseek-r1-tool-calling:7b",query)

    tool_call("llama3.1",query)
    tool_call("MFDoom/deepseek-r1-tool-calling:7b",query)

    approval("llama3.1",query)
    approval("MFDoom/deepseek-r1-tool-calling:7b",query)