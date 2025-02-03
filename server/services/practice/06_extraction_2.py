#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-28
# @function: 从文本中提取结构化信息
# @version : V0.5
# @Description ：从文本中提取结构化信息。

# https://python.langchain.com/docs/tutorials/extraction/


# Reference examples

from langchain_ollama import ChatOllama

def reference(model_name):
    messages = [
        {"role": "user", "content": "2 🦜 2"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "2 🦜 3"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "3 🦜 4"},
    ]

    response = ChatOllama(model=model_name,temperature=0.5,verbose=True).invoke(messages)
    return response.content

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

from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
    (
        "Fiona traveled far from France to Spain.",
        Person(name="Fiona", height_in_meters=None, hair_color=None),
    ),
    (
        "Alan Smith is 1.83 meters tall and has blond hair.",
        Person(name="Alan Smith", height_in_meters=1.83, hair_color="blond"),
    ),
]


messages = []

for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

for message in messages:
    message.pretty_print()

def extract(model_name,text):
    structured_llm = ChatOllama(model=model_name,temperature=0.5,verbose=True).with_structured_output(schema=Person)  
    response = structured_llm.invoke([text])
    return response

def extract_with_messages(model_name,text):
    structured_llm = ChatOllama(model=model_name,temperature=0.5,verbose=True).with_structured_output(schema=Person)  
    structured_llm.invoke(messages + [text])
    return response

if __name__ == '__main__':
    '''
    response = reference("llama3.1")
    print(f'\n llama3.1 response:\n{response}')

    response = reference("MFDoom/deepseek-r1-tool-calling:7b")
    print(f'\n deepseek-r1 response:\n{response}')
    '''
    # llama3.1无法自动把feet转换成meter，所以我们把这个问题简化了一些，在text中直接用meter做单位。
    text = {"role": "user", "content":"Roy is 1.73 meters tall and has black hair."}
    response = extract("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')
    response = extract_with_messages("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')


    text = {"role": "user", "content":"John Doe is 1.72 meters tall and has brown hair."}
    response = extract("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')
    response = extract_with_messages("llama3.1",text)
    print(f'\n llama3.1 response:\n{response}')
