#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-11
# @function: 使用tool的简单RAG
# @version : V0.5
# @Description ：去掉answer_style后回答比较好，有这个参数时回答很离谱。

# https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
    ),
]

vectorstore = InMemoryVectorStore.from_documents(
    documents, embedding=embedding_model
)

embedding = embedding_model.embed_query("dogs")
r = vectorstore.similarity_search_by_vector(embedding,k=1)
print(f'similarity_search_by_vector:{r}')

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
You are an assistant for question-answering tasks.
Use the below context to answer the question. If
you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer
concise.

Answer the question.

Question: {question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(f'json_schema:{rag_chain.input_schema.model_json_schema()}')

rag_tool = rag_chain.as_tool(
    name="pet_expert",
    description="Get information about pets.",
)

from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, [rag_tool])

for chunk in agent.stream(
    {"messages": [("human", "What would a pirate say dogs are known for?")]}
):
    print(chunk)
    print("----")
