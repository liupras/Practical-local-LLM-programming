#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-11
# @function: ad_hoc tool
# @version : V0.5


# https://python.langchain.com/docs/how_to/tools_prompting/

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1",temperature=0.1,verbose=True)
"""
temperature：用于控制生成语言模型中生成文本的随机性和创造性。
当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性;
当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
"""

# Create a tool
from langchain_core.tools import tool


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y


@tool
def add(x: int, y: int) -> int:
    "Add two numbers."
    return x + y


tools = [multiply, add]

# Let's inspect the tools
for t in tools:
    print("--")
    print(t.name)
    print(t.description)
    print(t.args)

# Creating our prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description

rendered_tools = render_text_description(tools)
print(rendered_tools)

system_prompt = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

chain = prompt | llm
message = chain.invoke({"input": "what's 3 plus 1132"})

# Let's take a look at the output from the model
# if the model is an LLM (not a chat model), the output will be a string.
if isinstance(message, str):
    print(f'LLM response: {message}')  
else:  # Otherwise it's a chat model
    print(f'chat model response: {message.content}') 

# Adding an output parse

from langchain_core.output_parsers import JsonOutputParser

chain = prompt | llm | JsonOutputParser()
message =chain.invoke({"input": "what's thirteen times 4"})
print(f'thirteen times 4: {message}')  

# Invoking the tool
# The function will select the appropriate tool by name, and pass to it the arguments chosen by the mode

from typing import Any, Dict, Optional, TypedDict

from langchain_core.runnables import RunnableConfig


class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]


def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

r = invoke_tool({"name": "multiply", "arguments": {"x": 3, "y": 5}})
print(f'test tool:{r}')

chain = prompt | llm | JsonOutputParser() | invoke_tool
result =chain.invoke({"input": "what's thirteen times 4.14137281"})
print(f'thirteen times 4.14137281:{result}')

from langchain_core.runnables import RunnablePassthrough

chain = (
    prompt | llm | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)
)
result = chain.invoke({"input": "what's thirteen times 4.14137281"})
print(f'thirteen times 4.14137281:{result}')