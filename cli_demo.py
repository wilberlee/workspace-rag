# -*- coding: utf-8 -*-
# @Time: 2024/5/29 11:06
# @File: cli_demo.py
# Desc:
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from embedding_module.model_config import *
import numpy as np
import json
import io
import csv
from cal_similarity import cal_similarities

app = FastAPI()

# TODO 将向量模型docker化，常启动，减少运行加载时间
embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict["text2vec"],
            model_kwargs={"device": EMBEDDING_DEVICE},
        )


def calculate_similarity_to_list(text, texts):
    """
    计算一个字符串与多个字符串之间的余弦相似度0~1
    :param text:
    :param texts:
    :return:
    """
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents(texts)
    cos_res = cal_similarities(np.array(query_result), np.array(doc_result))
    return cos_res


if __name__ == "__main__":
    while True:
        sentence = input("输入：")
        # sentence = "客户咨询申请三相电不成功的原因"
        sentence_list = ["今天天气真热", "三相电为什么不成功"]
        print(calculate_similarity_to_list(sentence, sentence_list))