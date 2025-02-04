#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-04
# @function: 用langgraph实现的chatbot
# @version : V0.5

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


def build_app_with_prompt_1(model_name):
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)

    def call_model(state: MessagesState):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You talk like a pirate. Answer all questions to the best of your ability.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt_template.invoke(state)        
        response = model.invoke(prompt)
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

from langchain_core.messages import HumanMessage

def test_app_1(model_name):
    app = build_app_with_prompt_1(model_name)

    config = {"configurable": {"thread_id": "abc345"}}
    query = "Hi! I'm Jim."

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())

    query = "What is my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())

from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

#  added a new language input to the prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def build_app_with_prompt_2(model_name):
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)

    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def test_app_2(model_name):
    app = build_app_with_prompt_2(model_name)

    config = {"configurable": {"thread_id": "abc456"}}
    language = "简体中文"

    query = "嘿，你好，我是刘大山。"    

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

    query = "我叫什么名字？"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages},
        config,
    )
    print(output["messages"][-1].pretty_print())

if __name__ == '__main__':
    mode_name = "llama3.1"
    #test_app_1(mode_name)
    test_app_2(mode_name)

    mode_name = "deepseek-r1"
    #test_app_1(mode_name)
    test_app_2(mode_name)