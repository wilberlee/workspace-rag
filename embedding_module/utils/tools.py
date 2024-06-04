#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/5/17 16:29
# @Author: lbl
from typing import List

import nltk
from langchain.document_loaders import UnstructuredFileLoader
from tqdm import tqdm

from configs.model_config import *
from utils.logger import logger
from textsplitter import ChineseTextSplitter

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

DEVICE_ = EMBEDDING_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_


def generate_prompt(
        related_docs: List[str],
        query: str,
        knowledge_graph_prompt: str = "",
        graph_metadata: dict = dict(),
        prompt_template: str = PROMPT_TEMPLATE
) -> str:
    """
    将知识库相关文本与预先设计的promt结构融合
    :param related_docs:
    :param query:
    :param prompt_template:
    :return:
    """
    context = "\n".join([doc.page_content for doc in related_docs])
    if graph_metadata != {}:
        context += "\n" + knowledge_graph_prompt
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def seperate_list(ls: List[int]) -> List[List[int]]:
    """
    连续索引分组，例[1,2,5,6]分组为[[1,2],[5,6]]
    :param ls:
    :return:
    """
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def load_file(filepath):
    """
    读取文档内容
    :param filepath:
    :return:
    """
    file_name = os.path.basename(filepath)
    file_name = file_name.replace(".", "_")

    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, filename=file_name)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, filename=file_name)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def texts_to_docs(filepath):
    """
    将文档加载切片转成docs列表
    :param filepath: 单文档路径或者文档目录
    :return: [Document(page_content, metadata),]
    """
    loaded_files = []
    docs = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            logger.error("{} 路径不存在".format(filepath))
            return None
        # 处理文件
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                # 文件切个 可优化todo
                docs = load_file(filepath)
                logger.info("{} 已成功加载".format(file))
                loaded_files.append(filepath)
            except Exception as e:
                logger.error("{} 未能成功加载：{}".format(file, e), exc_info=True)
                return None
        # 处理目录
        elif os.path.isdir(filepath):
            docs = []
            for file in tqdm(os.listdir(filepath), desc="加载文件"):
                fullfilepath = os.path.join(filepath, file)
                try:
                    docs += load_file(fullfilepath)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    logger.error("{} 未能成功加载：{}".format(file, e), exc_info=True)
    else:
        docs = []
        for file in filepath:
            try:
                docs += load_file(file)
                logger.info("{} 已成功加载".format(file))
                loaded_files.append(file)
            except Exception as e:
                logger.error("{} 未能成功加载：{}".format(file, e), exc_info=True)
    return docs

