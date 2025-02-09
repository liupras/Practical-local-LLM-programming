#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-08
# @function: 扩展的JsonOutputParser类。
# @version : V0.5
# @Description ：扩展的JsonOutputParser类，用来支持langchain支持的不好的模型。

from langchain_core.outputs import Generation
from langchain_core.output_parsers import JsonOutputParser
from typing import Any
import re

class ThinkJsonOutputParser(JsonOutputParser):
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。支持deepseek。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析 partial JSON 对象。
                If True, 输出将是一个 JSON 对象，其中包含迄今为止已返回的所有键。
                If False, 输出将是完整的 JSON 对象。
                默认值为 False.

        Returns:
            解析后的 JSON 对象。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """

        text = result[0].text
        text = text.strip()

        # 判断是否为 deepseek生成的内容，如果是的话，提取其中的json字符串
        if '<think>' in text and '</think>' in text:  
            match = re.search(r'\{.*\}', text.strip(), re.DOTALL)
            if match:
                text = match.group(0)
        result[0].text = text

        return super().parse_result(result, partial=partial)