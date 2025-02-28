#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-27
# @function: 流式输出
# @version : V0.5
# @Description ：测试流式输出。

llm_model_name = "qwen2.5"

"""
1. 工具
"""

import random
from langchain_core.tools import tool

@tool
def where_cat_is_hiding() -> str:
    """Where is the cat hiding right now?"""
    return random.choice(["under the bed", "on the shelf"])


@tool
def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "bed" in place:  # For under the bed
        return "socks, shoes and dust bunnies"
    if "shelf" in place:  # For 'shelf'
        return "books, penciles and pictures"
    else:  # if the agent decides to ask about a different place
        return "cat snacks"
    
"""
2. 初始化智能体
"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant"),
  ("placeholder", "{chat_history}"),
  ("human", "{input}"),
  ("placeholder", "{agent_scratchpad}"),
])

tools = [get_items, where_cat_is_hiding]

from langchain.agents import create_tool_calling_agent
from langchain_ollama import ChatOllama

from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

class CustomStreamingHandler(FinalStreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def on_llm_new_token(self, token: str, **kwargs):
        """每当 LLM 生成新 token 时调用"""
        self.buffer.append(token)
        print(token, end="^", flush=True)  # 直接流式输出最终结果

import sys
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class CallbackHandler(StreamingStdOutCallbackHandler):
    """自定义输出"""
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        """智能体会逐渐返回json格式的结果，这里只输出 action_input 的内容"""
        self.content += token
        if "Final Answer" in self.content:
            # 现在我们到了 Final Answer 部分，但不要打印。
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:

                # 当字符串中包含 '}' 时，移除 '}' 和后面的字符。
                index = token.find('}')  # 找到 '}' 的索引
                if index != -1:
                    self.final_answer = False
                    token = token[:index]

                sys.stdout.write(token) 
                if index == -1:
                    sys.stdout.write('^')
                sys.stdout.flush()

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True,callbacks=[StreamingStdOutCallbackHandler()])

agent = create_tool_calling_agent(model, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

def ask(question):
    """咨询智能体"""
    #chunks = []
    for chunk in agent_executor.stream({"input": question}):
        #chunks.append(chunk)
        print_normal(chunk)

def print_normal(chunk):
    print(chunk)
    print("----")

def print_simple(chunk):
    # Note: We use `pprint` to print only to depth 1, it makes it easier to see the output from a high level, before digging in.
    import pprint
    print("----")
    pprint.pprint(chunk, depth=1)

def print_useful(chunk):
    # Agent Action
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
    # Observation
    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"Tool Result: `{step.observation}`")
    # Final result
    elif "output" in chunk:
        print(f'Final Output: {chunk["output"]}')
    else:
        raise ValueError()
    print("---")

async def ask_2(question):
    async for event in agent_executor.astream_events(
        {"input": question},
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")


agent_executor_stream = AgentExecutor(agent=agent, tools=tools,verbose=False,callbacks=[CallbackHandler()]).with_config(
    {"run_name": "Agent_stream"}
)

#agent_executor_stream.agent.llm.callbacks =[CallbackHandler()]

def ask_3(question):
    """咨询智能体"""
    agent_executor_stream.invoke({"input": question})
    '''
    for chunk in agent_executor_stream.stream({"input": question}):
        pass
        #print(chunk)
    '''
    

if __name__ == '__main__':

    #place = where_cat_is_hiding.invoke({})
    #items = get_items.invoke({"place": "shelf"})

    #ask("what's items are located where the cat is hiding?")

    '''
    import asyncio
    asyncio.run(ask_2("where is the cat hiding? what items are in that location?"))
    '''

    #ask_3("what's items are located where the cat is hiding?")
    ask_3("请参考哪吒闹海的故事架构，写一篇200-300字的神话故事。")