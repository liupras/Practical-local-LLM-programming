#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-19
# @function: 用langgraph实现的rag
# @version : V0.5

# https://python.langchain.com/docs/tutorials/qa_chat_history/

import os 
os.environ['USER_AGENT'] = 'rag_graph_2'

"""
确定文件路径
"""

import sys

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

def get_persist_directory(model_name):
    """矢量数据库存储路径"""
    model_name = model_name.replace(":","-")
    return os.path.join(current_dir,f'assert/animals_{model_name}')

"""
1. 创建矢量数据库对象
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embed_model_name = "shaw/dmeta-embedding-zh"
vector_store = Chroma(persist_directory=get_persist_directory(embed_model_name),embedding_function=OllamaEmbeddings(model=embed_model_name))

"""
2. 实现Langgraph 链
"""

from langchain_core.tools import tool

@tool(response_format="content_and_artifact",parse_docstring=True)      # docstring的内容对agent自动推理影响比较大
def retrieve(query: str):
    """检索与 query参数内容 相关的信息

    Args:
        query: 要搜索的字符串。 
    """

    print(f"start retrieve:{query}")

    # 定义相似度阈值。因为这种相似性检索并不考虑相似性大小，如果不限制可能会返回相似性不大的文档， 可能会影响问答效果。
    similarity_threshold = 0.6
    retrieved_docs = vector_store.similarity_search_with_score(query, k=3)

    # 根据相似度分数过滤结果
    filtered_docs = [
        doc for doc, score in retrieved_docs if score <= similarity_threshold
    ]

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in filtered_docs
    )

    if not serialized:
        return "抱歉，我找不到任何相关信息。", None
    else:
        return serialized, filtered_docs
    

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from langgraph.graph import MessagesState, StateGraph,END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def build_graph_with_memory(llm_model_name):
    """构建 langgraph 链"""
    
    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)

    # 1: 生成可能包含要发送的工具调用的 AIMessage。
    def query_or_respond(state: MessagesState):
        """生成用于检索或响应的工具调用。"""

        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        """
        这里会自动进行指代消解：根据上下文自动修改问题，把问题中的代词替换成上下文中的内容
        """
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"messages": [response]}

    # 2: 执行检索
    tools = ToolNode([retrieve])

    # 3: 使用检索到的内容生成响应。
    def generate(state: MessagesState):
        """生成回答。"""

        # 获取生成的 ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # 获取 ToolMessages 的内容，并格式化为提示词
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "你是一个负责答疑任务的助手。 "
            "使用以下检索到的上下文来回答问题。 "
            "如果你不知道答案，就说你不知道。 "
            "最多使用三句话并保持答案简洁。 "
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

        # 执行
        response = llm.invoke(prompt)
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"messages": [response]}

    # 4: 串联节点和边，构建图
    graph_builder = StateGraph(MessagesState)

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

    # 增加记忆功能
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


def ask_with_history(graph,thread_id,question):
    """提问，记录聊天历史"""

    print('---ask_with_history---')
    conf = {"configurable": {"thread_id": thread_id}}
    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config = conf,
        ):
            step["messages"][-1].pretty_print()


"""
3. 智能体
"""

"""
智能体利用 LLM 的推理能力在执行过程中做出决策。使用代理可以让您在检索过程中减轻额外的判断力。
虽然它们的行为比上述“链”更难预测，但它们能够执行多个检索步骤来处理查询，或者在单个搜索中进行迭代。
"""

from langgraph.prebuilt import create_react_agent

def create_agent(llm_model_name):
    """创建智能体"""

    llm = ChatOllama(model=llm_model_name,temperature=0, verbose=True)
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools=[retrieve], checkpointer=memory)
    return agent_executor


def ask_agent(agent,thread_id,question):
    """咨询智能体"""

    print('---ask_agent---')
    conf = {"configurable": {"thread_id": thread_id}}
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=conf,
    ):
        step["messages"][-1].pretty_print()

def show_graph():
    """图形化显示链和智能体结构"""

    from utils import show_graph

    graph = build_graph_with_memory("qwen2.5")
    show_graph(graph)

    agent = create_agent("qwen2.5")
    show_graph(agent)


def test_model(llm_model_name):
    """测试大语言模型"""

    print(f'------{llm_model_name}------')

    question1 = "羊的学名是什么？"
    question2 = "它有什么特点？"
    thread_id = "liu2233"

    graph = build_graph_with_memory(llm_model_name)
    ask_with_history(graph,thread_id,question1)
    ask_with_history(graph,thread_id,question2)

    agent = create_agent(llm_model_name)
    ask_agent(agent,thread_id,question1)
    ask_agent(agent,thread_id,question2)

if __name__ == '__main__':
    
    test_model('qwen2.5')
    test_model('llama3.1')
    test_model("MFDoom/deepseek-r1-tool-calling:7b")

    show_graph()