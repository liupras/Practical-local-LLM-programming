#coding=utf-8

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-07
# @Description: 利用本地大模型测试矢量数据库
# @version : V0.5

from langchain_core.documents import Document

# 使用 nomic-embed-text 做嵌入检索很好，使用 llama3.1 效果一般
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# 使用 nomic-embed-text 做嵌入不行，使用 llama3.1 效果还行。
documents_zh = [
    Document(
        page_content="狗是极好的伙伴，以忠诚和友好而闻名。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="猫是独立的宠物，它们通常喜欢拥有自己的空间。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="金鱼是适合初学者饲养的宠物，需要的护理相对简单。",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="兔子是群居动物，需要足够的空间来跳跃。",
        metadata={"source": "mammal-pets-doc"},
    ),
]

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# nomic-embed-text  llama3.1    EntropyYue/chatglm3
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
"""
nomic-embed-text: 一个高性能开放嵌入模型，只有27M，具有较大的标记上下文窗口。
在做英文的嵌入和检索时，明显比llama3.1要好，可惜做中文不行。
"""    

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.3,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""
 
vectorstore = Chroma.from_documents(
    documents_zh,
    embedding=embeddings_model,
)

def search():
    """
    矢量检索
    """

    print("search test...")

    # Return documents based on similarity to a string query
    r = vectorstore.similarity_search("cat")
    print(f'similarity_search:{r}')

    # Return scores
    r = vectorstore.similarity_search_with_score("cat")
    print(f'similarity_search_with_score:{r}')

    # Return documents based on similarity to an embedded query
    embedding = embeddings_model.embed_query("cat")
    vectorstore.similarity_search_by_vector(embedding)


from langchain_core.runnables import RunnableLambda

def retriever():
    """
    retriever测试
    vectorstore不是Runnable对象，所以不能集成到LangChain中，只能单独调用；Retrievers就是Runnable对象了，可以集成到LangChain。
    """

    print("retriever test...")

    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
    r = retriever.batch(["cat", "shark"])
    print(f'retriever.batch:{r}')

    # as_retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    r= retriever.batch(["cat", "shark"])
    print(f'as_retriever.batch:{r}')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def RAG(question):
    """
    RAG测试
    """
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    response = rag_chain.invoke({"question": question})

    # print(response.content)
    return response.content

if __name__ == '__main__':

    #search()
    #retriever()

    # 使用 nomic-embed-text 可以正确处理，使用llama3.1 失败。
    print(RAG("tell me about cats"))

