#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-28
# @function: 从文本中提取结构化信息
# @version : V0.5
# @Description ：从文本中提取结构化信息。

# https://python.langchain.com/docs/tutorials/extraction/


# The Schema

from typing import Optional
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[float] = Field(
        default=None, description="Height measured in meters"
    )

# The Extractor

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

from langchain_ollama import ChatOllama

def single_entry(model_name,text):

    structured_llm = ChatOllama(model=model_name,temperature=0.5,verbose=True).with_structured_output(schema=Person)

    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    return response

from typing import List

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

def multiple_entry(model_name,text):
    structured_llm = ChatOllama(model=model_name,temperature=0.5,verbose=True).with_structured_output(schema=Data)  
    response = structured_llm.invoke([text])
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    return response


if __name__ == '__main__':

    # llama3.1无法自动把feet转换成meter，所以我们把这个问题简化了一些，在text中直接用meter做单位。
    text = "Alan Smith is 1.83 meters tall and has blond hair."
    response = single_entry("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')

    response = single_entry("MFDoom/deepseek-r1-tool-calling:7b",text)
    print(f'\n deepseek response:\n{response}')

    text = "Alan Smith is 6 feet tall and has blond hair."
    response = single_entry("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')

    response = single_entry("MFDoom/deepseek-r1-tool-calling:7b",text)
    print(f'\n deepseek response:\n{response}')

    
    text = "Alan Smith is 1.83 meters tall and has blond hair. John Doe is 1.72 meters tall and has brown hair."
    response = multiple_entry("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')

    response = multiple_entry("MFDoom/deepseek-r1-tool-calling:7b",text)
    print(f'\n deepseek response:\n{response}')
 
    text = "Alan Smith is 1.88 meters tall and has blond hair. John Doe is 7 feet tall and has brown hair."
    response = multiple_entry("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')

    response = multiple_entry("MFDoom/deepseek-r1-tool-calling:7b",text)
    print(f'\n deepseek response:\n{response}')
    