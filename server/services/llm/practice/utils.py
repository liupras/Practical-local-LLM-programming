#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-20
# @function: 工具集
# @version : V0.5
# @Description ：工具集。

def show_graph(graph):
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()