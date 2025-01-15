#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-15
# @function: 用langgraph实现的chatbot
# @version : V0.5

from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1",temperature=0.3,verbose=True)

from langchain_core.messages import HumanMessage

def chat_with_no_memory():
    response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    print(f'chat_with_no_memory:{response.content}')

    # We can see that it doesn't take the previous conversation turn into context, and cannot answer the question. This makes for a terrible chatbot experience!
    response = model.invoke([HumanMessage(content="What's my name?")])
    print(f'chat_with_no_memory 2:{response.content}')

from langchain_core.messages import AIMessage

def chat_with_memory():
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

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Message persistence
def message_persistence():

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

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
    #chat_with_memory()
    message_persistence()