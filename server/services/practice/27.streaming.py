#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-26
# @function: 流式输出
# @version : V0.5
# @Description ：测试流式输出。

"""
langchain 的流式输出
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,AIMessage

def chat(llm_model_name,question):
    """与大模型聊天，一次性输出"""
    model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True)
    response = model.invoke([HumanMessage(content=question)])
    print(f'AI:\n{response.content}')

def chat_stream(llm_model_name,question):
    """与大模型聊天，流式输出"""
    model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True)
    for chunk in model.stream([HumanMessage(content=question)]):
        if isinstance(chunk, AIMessage) and chunk.content !='':
            print(chunk.content,end="^")

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def chat_stream_2(llm_model_name,question):
    """与大模型聊天，流式输出"""
    model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True,callbacks=[StreamingStdOutCallbackHandler()])
    model.invoke([HumanMessage(content=question)])

class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    """自定义流式回调处理器，在流式输出时使用 ^ 作为分隔符"""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """重写方法，修改输出格式"""
        print(token, end="^", flush=True)  # 使用 `^` 作为分隔符

def chat_stream_3(llm_model_name,question):
    """与大模型聊天，流式输出"""
    model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True,callbacks=[CustomStreamingHandler()])
    model([HumanMessage(content=question)])

"""
智能体的流式输出
"""

# 初始化对话存储
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

llm_model_name = "qwen2.5"
model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True,callbacks=[CustomStreamingHandler()])

from langchain_community.agent_toolkits.load_tools import load_tools

# 创建一个工具来观察它如何影响流的输出
tools = load_tools(["llm-math"], llm=model)

from langchain.agents import AgentType, initialize_agent

# 创建智能体
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=model,
    memory=memory,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    return_intermediate_steps=False
)

def chat_agent(quesion):
    """与智能体聊天，它会把所有内容都流式输出"""
    agent.invoke(quesion)

def chat_agent_2(quesion):
    """与智能体聊天，它会把所有内容都流式输出"""
    from langchain.callbacks.streaming_stdout_final_only import (
        FinalStreamingStdOutCallbackHandler,
    )

    agent.agent.llm_chain.llm.callbacks = [
        FinalStreamingStdOutCallbackHandler(
            answer_prefix_tokens=["Final", "Answer"]   # 流式输出 Final 和 Answer 后面的内容 
        )
    ]

    agent.invoke(quesion)

import sys

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

def chat_agent_3(quesion):
    """与智能体聊天，它会把所有内容都流式输出"""
    agent.agent.llm_chain.llm.callbacks =[CallbackHandler()]
    agent.invoke(quesion)


if __name__ == '__main__':

    question = "中国有多少个地级市？"
    chat("qwen2.5",question)
    chat_stream("qwen2.5",question)
    chat_stream_2("qwen2.5",question)
    chat_stream_3("qwen2.5",question)

    chat_agent(question)
    chat_agent("9的平方是多少？")
    chat_agent_2("9的平方是多少？")
    chat_agent_3("9的平方是多少？")



