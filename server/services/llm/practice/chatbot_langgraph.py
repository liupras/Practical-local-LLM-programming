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

# Message persistence
def message_persistence():

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


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
   
def message_persistence_with_prompt():

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

def message_persistence_with_prompt_2():

    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc456"}}
    language = "简体中文"

    query = "Hi! I'm Bob."    

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

    query = "What is my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages},
        config,
    )
    print(output["messages"][-1].pretty_print())

from langchain_core.messages import SystemMessage, trim_messages

def trim_message_history():
    """
    Importantly, you will want to do this BEFORE the prompt template but AFTER you load previous messages from Message History.
    """
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    messages_trimed = trimmer.invoke(messages)
    print(f'messages_trimed:{messages_trimed}')

    def call_model(state: State):
        """
        Importantly, you will want to do this BEFORE the prompt template but AFTER you load previous messages from Message History.
        """
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke(
            {"messages": trimmed_messages, "language": state["language"]}
        )
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc567"}}
    query = "What is my name?"
    language = "简体中文"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

    config = {"configurable": {"thread_id": "abc678"}}
    query = "What math problem did I ask?"
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

if __name__ == '__main__':
    #chat_with_memory()
    #message_persistence()
    #message_persistence_with_prompt()
    #message_persistence_with_prompt_2()
    trim_message_history()