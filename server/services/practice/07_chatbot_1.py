#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-04
# @function: 用langgraph实现的chatbot
# @version : V0.5

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def chat(model_name):
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)
    response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    print(f'chat_with_no_memory:{response.content}')

    # We can see that it doesn't take the previous conversation turn into context, and cannot answer the question. This makes for a terrible chatbot experience!
    response = model.invoke([HumanMessage(content="What's my name?")])
    print(f'chat_with_no_memory 2:{response.content}')

from langchain_core.messages import AIMessage

def chat_with_memory(model_name):
    '''具有记忆功能'''
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)
    response = model.invoke(
        [
            HumanMessage(content="Hi! I'm Bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )
    print(f'chat_with_memory:{response.content}')

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Message persistence
def build_app(model_name):

    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)

    # Define the function that calls the model
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def message_persistence(model_name):
    app = build_app(model_name)

    config = {"configurable": {"thread_id": "abc123"}}

    query = "Hi! I'm Bob."

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())  # output contains all messages in state

    query = "What's my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())

    # different thread_id
    config = {"configurable": {"thread_id": "abc234"}}

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())

    config = {"configurable": {"thread_id": "abc123"}}

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    print(output["messages"][-1].pretty_print())

if __name__ == '__main__':
    mode_name = "llama3.1"
    print(f'----------------------------{mode_name}---------------------------')
    chat(mode_name)
    chat_with_memory(mode_name)

    message_persistence(mode_name)

    mode_name = "deepseek-r1"
    print(f'----------------------------{mode_name}---------------------------')
    chat(mode_name)
    chat_with_memory(mode_name)

    message_persistence(mode_name)