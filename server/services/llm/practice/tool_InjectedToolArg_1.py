#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 防止LLM生成某些参数
# @version : V0.5
# @Description ：防止LLM生成某些参数。

# https://python.langchain.com/docs/how_to/tool_runtime/

"""
您可能需要将仅在运行时才知道的值绑定到工具。例如，工具逻辑可能需要使用发出请求的用户的 ID。
大多数情况下，此类值不应由 LLM 控制。事实上，允许 LLM 控制用户 ID 可能会导致安全风险。
相反，LLM 应该只控制本应由 LLM 控制的工具参数，而其他参数（如用户 ID）应由应用程序逻辑固定。
本操作指南向您展示了如何防止模型生成某些工具参数并在运行时直接注入它们。
"""

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

# Hiding arguments from the model

from typing import List
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolArg, tool

user_to_pets = {}


@tool(parse_docstring=True)
def update_favorite_pets(
    pets: List[str], user_id: Annotated[str, InjectedToolArg]
) -> None:
    """Add the list of favorite pets.

    Args:
        pets: List of favorite pets to set.
        user_id: User's ID.
    """
    print(f'update_favorite_pets is called:{user_id}')
    user_to_pets[user_id] = pets


@tool(parse_docstring=True)
def delete_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """Delete the list of favorite pets.

    Args:
        user_id: User's ID.
    """
    print(f'delete_favorite_pets is called:{user_id}')
    if user_id in user_to_pets:
        del user_to_pets[user_id]


@tool(parse_docstring=True)
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """List favorite pets if any.

    Args:
        user_id: User's ID.
    """
    print(f'list_favorite_pets is called:{user_id}')
    return user_to_pets.get(user_id, [])

# If we look at the input schemas for these tools, we'll see that user_id is still listed:
print(f'get_input_schema:{update_favorite_pets.get_input_schema().model_json_schema()}')

# But if we look at the tool call schema, which is what is passed to the model for tool-calling, user_id has been removed:
print(f'tool_call_schema:{update_favorite_pets.tool_call_schema.model_json_schema()}')

# So when we invoke our tool, we need to pass in user_id:
user_id = "123"
update_favorite_pets.invoke({"pets": ["lizard", "dog"], "user_id": user_id})
print(f'user_to_pets:{user_to_pets}')
print(f'list_favorite_pets.invoke:{list_favorite_pets.invoke({"user_id": user_id})}')

# But when the model calls the tool, no user_id argument will be generated:
tools = [
    update_favorite_pets,
    delete_favorite_pets,
    list_favorite_pets,
]
llm_with_tools = llm.bind_tools(tools)

query = "my favorite animals are cats and parrots"
#query = "what's your name?"

# 将文本转化为json结构
print("---1、调用LLM，将请求转化为json结构---")
ai_msg = llm_with_tools.invoke(query)
print(f'result:{ai_msg.tool_calls}')

# Injecting arguments at runtime
from copy import deepcopy

from langchain_core.runnables import chain

@chain
def inject_user_id(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["user_id"] = user_id
        tool_calls.append(tool_call_copy)
    return tool_calls

new_args = inject_user_id.invoke(ai_msg)
print(f'inject_user_id:{new_args}')

# And now we can chain together our model, injection code, and the actual tools to create a tool-executing chain:
tool_map = {tool.name: tool for tool in tools}

@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]

# And now we can chain together our model, injection code, and the actual tools to create a tool-executing chain:
chain = llm_with_tools | inject_user_id | tool_router.map()

# 调用chain
print("--通过chain实现：2、注入新参数user_id；3、直接调用tool生成结果；4、调用LLM，生成流畅的答案---")
result = chain.invoke(query)
print(f'chain.invoke:{result}')
print(f'now user_to_pets :{user_to_pets}')