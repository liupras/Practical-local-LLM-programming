#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: 用langgraph实现的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/qa_chat_history/

import os 
os.environ['USER_AGENT'] = 'rag_graph_2'

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0, verbose=True)

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vector store
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Chains
from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool


@tool(response_format="content_and_artifact",parse_docstring=True)
def retrieve(keywords: str):
    """Retrieve information related to keywords.

    Args:
        keywords: keywords to be searched. 
    """

    print(f"start retrieve key_content:{keywords}")
    # 定义相似度阈值。因为这种相似性检索并不考虑相似性大小，如果不限制可能会返回相似性不大的文档，影响问答效果。
    similarity_threshold = 0.6
    retrieved_docs = vector_store.similarity_search_with_score(keywords, k=8)
    """
    使用llama3.1测试时，发现：
    在提问第二个问题：Can you look up some common ways of doing it？时，前两个返回的结果与第一个问题是相同的，
    实际上与此问题最相关的结果是score偏低的第三个结果。
    所以在大模型没有OpenAI那么强的情况下，返回的文档多一点比较保险。
    """

    # 根据相似度分数过滤结果
    filtered_docs = [
        doc for doc, score in retrieved_docs if score >= similarity_threshold
    ]

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in filtered_docs
    )

    if not serialized:
        return "Sorry, I could not find any relevant information.", None
    else:
        return serialized, filtered_docs

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    """
    这里会自动进行指代消解：根据上下文自动修改问题，把问题中的代词替换成上下文中的内容
    """
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

def show_graph():
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()

def ask(question):
    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph_with_memory = graph_builder.compile(checkpointer=memory)

def show_graph_with_memory():
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph_with_memory.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()

def ask_with_history(question,thread_id):

    # Specify an ID for the thread
    config = {"configurable": {"thread_id": thread_id}}

    for step in graph_with_memory.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()

#agents
"""
Agents leverage the reasoning capabilities of LLMs to make decisions during execution. Using agents allows you to offload additional discretion over the retrieval process. 
Although their behavior is less predictable than the above "chain", they are able to execute multiple retrieval steps in service of a query, or iterate on a single search.
"""
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools=[retrieve], checkpointer=memory)

def show_agent_with_memory():
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = agent_executor.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()

def ask_agent(question,thread_id):

    # Specify an ID for the thread
    config = {"configurable": {"thread_id": thread_id}}

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    #show_graph()
    #ask("Hello world")
    #ask("What is Task Decomposition?")
    #show_graph_with_memory()
    thread_id = '12345'
    #ask_with_history("What is Task Decomposition?",thread_id)
    #ask_with_history("Can you look up some common ways of doing it?",thread_id)
    #show_agent_with_memory()

    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    ask_agent(input_message,thread_id)
