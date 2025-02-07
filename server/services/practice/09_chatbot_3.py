#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-06
# @function: 用langgraph实现的chatbot
# @version : V0.5

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage, trim_messages

def get_trimmer(model_name,max_tokens):
    """
    重要：请务必在在加载之前的消息之后，并且在提示词模板之前使用它。
    """
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)
    trimmer = trim_messages(
        max_tokens=max_tokens,  #设置裁剪后消息列表中允许的最大 token 数量
        strategy="last",        #指定裁剪策略为保留最后的消息，即从消息列表的开头开始裁剪，直到满足最大 token 数量限制。
        token_counter=model,    #通过model来计算消息中的 token 数量。
        include_system=True,    #在裁剪过程中包含系统消息（SystemMessage）
        allow_partial=False,    #不允许裁剪出部分消息，即要么保留完整的消息，要么不保留，不会出现只保留消息的一部分的情况。
        start_on="human",   #从人类消息（HumanMessage）开始进行裁剪，即裁剪时会从第一个HumanMessage开始计算 token 数量，之前的系统消息等也会被包含在内进行整体裁剪考量。
    )
    return trimmer

messages = [
    SystemMessage(content="你是个好助手"),
    HumanMessage(content="你好，我是刘大钧"),
    AIMessage(content="你好"),
    HumanMessage(content="我喜欢香草冰淇淋"),
    AIMessage(content="很好啊"),
    HumanMessage(content="3 + 3等于几？"),
    AIMessage(content="6"),
    HumanMessage(content="谢谢"),
    AIMessage(content="不客气"),
    HumanMessage(content="和我聊天有意思么？"),
    AIMessage(content="是的，很有意思"),
]

def test_trimmer(model_name,max_tokens):
    t = get_trimmer(model_name,max_tokens)
    messages_trimed = t.invoke(messages)
    print(f'{model_name} messages_trimed:\n{messages_trimed}')

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#  added a new language input to the prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个乐于助人的助手。请用{language}尽力回答所有问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Sequence
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

def build_app(model_name,max_tokens):
    model = ChatOllama(model=model_name,temperature=0.3,verbose=True)

    def call_model(state: State):
        trimmer = get_trimmer(model_name=model_name,max_tokens=max_tokens)
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
    return app

def test_app(model_name,max_tokens):
    app = build_app(model_name,max_tokens)

    config = {"configurable": {"thread_id": "abc456"}}
    language = "简体中文"

    query = "我叫什么名字？"    

    input_messages = messages + [HumanMessage(query)]

    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

    app = build_app(model_name,max_tokens)
    """
    重新构建app的目的是方便测试消息裁剪，否则app会缓存messages，导致下面的问题回答不出来。
    """

    query = "我问过什么数学问题？"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print(output["messages"][-1].pretty_print())

def stream(human_message,thread_id,model_name,max_tokens=140,language="简体中文"):
    '''
    流式输出
    '''
    app = build_app(model_name,max_tokens)
    for chunk, _ in app.stream(
        {"messages":[HumanMessage(content=human_message)],"language":language}, 
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            yield chunk.content

def test_1(model_name):
    max_token = 140
    test_trimmer(model_name,max_token)    
    test_app(model_name,max_token)

def test_2(model_name):
    max_token = 140
    thread_id = "liupras"
    query = "请以葛优的语气，写一首幽默的打油诗。"
    language = "简体中文"

    print(f"---------{model_name}---------------")

    for r in stream(query,thread_id,model_name,max_tokens=max_token ,language=language):
        if r is not None:            
            print (r, end="|")

if __name__ == '__main__':

    #test_1("llama3.1")
    #test_1("deepseek-r1")

    test_2("llama3.1")
    test_2("deepseek-r1")
    
    