#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 在Agent中增加人的干预
# @version : V0.5
# @Description ：在Agent中增加人的干预。

# https://python.langchain.com/docs/how_to/tools_human/

"""
有些工具我们不信任模型能够自行执行。在这种情况下，我们可以做的一件事是在调用工具之前要求人工批准。
"""

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.tools import tool


@tool
def count_emails(last_n_days: int) -> int:
    """Dummy function to count number of e-mails. Returns 2 * last_n_days."""
    print(f'count_emails is called:{last_n_days}')
    return last_n_days * 2


@tool
def send_email(message: str, recipient: str) -> str:
    """Dummy function for sending an e-mail."""
    print(f'send_email is called:{recipient}')
    return f"Successfully sent email to {recipient}."


tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls

# 使用chain调用tools很方便。这里直接输出json格式的结果，并未再次调用llm输出流畅的自然语言。
chain = llm_with_tools | call_tools
result = chain.invoke("how many emails did i get in the last 5 days?")
print(f'chain.invoked:{result}')

# Adding human approval

import json


class NotApproved(Exception):
    """Custom exception."""
    print(f'Not approved:{Exception}')


def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        print("human_approval is called,not approved.")
        raise NotApproved(f"Tool invocations not approved:\n\n{tool_strs}")
    print("human_approval is called,Approved.")
    return msg

# 这里直接输出json格式的结果，并未再次调用llm输出流畅的自然语言。
chain = llm_with_tools | human_approval | call_tools

try:
    result = chain.invoke("how many emails did i get in the last 5 days?")
    print(f'human-in-the-loop chain.invoke:{result}')
except NotApproved as e:
    print(f'Not approved:{e}')