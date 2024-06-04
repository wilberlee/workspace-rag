# @Description : fiass操作的接口，包括插入数据、删除数据、删除集合、查询
import os
from typing import List
import numpy as np

import faiss
from langchain import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from model_config import *
from utils import torch_gc


def load_collection(folder_path: str, embeddings: Embeddings, index_name: str = "index"):
    """
    加载本地集合
    :param folder_path: 知识库路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :return:
    """
    return FAISS.load_local(folder_path, embeddings, index_name)


def insert_data(folder_path: str, embeddings: Embeddings, docs: List[Document], index_name: str = "index"):
    """
    插入数据
    :param folder_path: 知识库路径
    :param embeddings: 向量化模型
    :param docs: 切分后的数据集
    :param index_name: 索引名称
    :return:
    """
    vs = load_collection(folder_path, embeddings, index_name)
    return vs.add_documents(docs)



def insert_data_from_documents(docs: List[Document], embeddings:Embeddings):
    """
    根据documents新建collection
    :param docs: 切分后的数据集
    :param embeddings: 向量化模型
    :return:
    """
    return FAISS.from_documents(docs, embeddings)



def insert_data_by_faiss(vs_path: str,
                         embeddings: Embeddings,
                         docs: List[Document],
                         catalogs: str = "",
                         index_name: str = "index"):
    """
    通过faiss插入数据
    :param vs_path:知识库路径
    :param embeddings: 向量化模型
    :param docs: 切分后的数据集
    :param catalogs: 知识库名称
    :param index_name: 索引名称
    :return:
    """
    # 如果当前知识库存在，则在当前知识库追加索引并保存，如果不存在则新建知识库写入索引文件
    if vs_path and os.path.isdir(vs_path):
        vector_store = load_collection(vs_path, embeddings, index_name)
        vector_store.add_documents(docs)
        # load_collection(vs_path, embeddings, index_name)
        # insert_data(vs_path, embeddings, docs, index_name)
    else:
        if not vs_path:
            vs_path = os.path.join(VS_ROOT_PATH, catalogs)
        vector_store = insert_data_from_documents(docs, embeddings)
    torch_gc()
    vector_store.save_local(vs_path)


def search_by_vector(vs_path: str,
                     embeddings: Embeddings,
                     index_name: str,
                     vector: List[float],
                     topk: int = 4):
    """
    根据vector查询faiss
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param vector: vector
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    return vs.similarity_search_by_vector(vector, topk)


def search_by_vectors(vs_path: str,
                      embeddings: Embeddings,
                      index_name: str,
                      vectors: List[List[float]],
                      topk: int = 4):
    """
    根据vectors查询faiss
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param vectors: vectors
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    result_lists = []
    for each in vectors:
        result_list = vs.similarity_search_by_vector(each, topk)
        for result in result_list:
            result_lists.append(result)
    return result_lists


def search_by_text(vs_path: str,
                   embeddings: Embeddings,
                   index_name: str,
                   query: str,
                   topk: int = 4):
    """
    根据query查询faiss
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param query: query
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    return vs.similarity_search(query, topk)


def search_by_texts(vs_path: str,
                    embeddings: Embeddings,
                    index_name: str,
                    querys: List[str],
                    topk: int = 4):
    """
    根据querys查询faiss
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param querys: querys
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    result_lists = []
    for each in querys:
        result_list = vs.similarity_search(each, topk)
        for result in result_list:
            result_lists.append(result)
    return result_lists


def search_with_score_by_vector(vs_path: str,
                                embeddings: Embeddings,
                                index_name: str,
                                vector: List[float],
                                topk: int = 4):
    """
    根据vector查询faiss,返回结果带得分
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param vector: vector
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    return vs.similarity_search_with_score_by_vector(vector, topk)


def search_with_score_by_vectors(vs_path: str,
                                 embeddings: Embeddings,
                                 index_name: str,
                                 vectors: List[List[float]],
                                 topk: int = 4):
    """
    根据vectors查询faiss,返回结果带得分
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param vectors: vectors
    :param topk: 返回数量
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    result_lists = []
    for each in vectors:
        result_list = vs.similarity_search_with_score_by_vector(each, topk)
        for result in result_list:
            result_lists.append(result)
    return result_lists


def search_with_score_by_text(vs_path: str,
                              embeddings: Embeddings,
                              index_name: str,
                              query: str,
                              similarity_search_function,
                              topk: int = 4,
                              chunk_size: int = CHUNK_SIZE):
    """
    根据query,返回结果带得分
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param query: query
    :param similarity_search_function: 重定义函数
    :param topk: 返回数量
    :param chunk_size: 最大长度
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    # faiss重定义similarity_search_with_score_by_vector
    FAISS.similarity_search_with_score_by_vector = (
        similarity_search_function
    )
    # 查询时的chunk_size，文本切分时，也有一个chunk_size，这两个不同的概念
    vs.chunk_size = chunk_size
    return vs.similarity_search_with_score(query, topk)


def search_with_score_by_texts(vs_path: str,
                               embeddings: Embeddings,
                               index_name: str,
                               querys: List[str],
                               similarity_search_function,
                               topk: int = 4,
                               chunk_size: int = CHUNK_SIZE):
    """
    根据querys,返回结果带得分
    :param vs_path: collection路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param querys: querys
    :param similarity_search_function: 重定义函数
    :param topk: 返回数量
    :param chunk_size: 最大长度
    :return:
    """
    vs = load_collection(vs_path, embeddings, index_name)
    # faiss重定义similarity_search_with_score_by_vector
    FAISS.similarity_search_with_score_by_vector = (
        similarity_search_function
    )
    # 查询时的chunk_size，文本切分时，也有一个chunk_size，这两个不同的概念
    vs.chunk_size = chunk_size
    result_lists = []
    for each in querys:
        result_list = vs.similarity_search_with_score(each, topk)
        for result in result_list:
            result_lists.append(result)
    return result_lists




def delete_data_by_faiss(catalog: str, filenames: str):
    """
    根据文件名删除对应索引
    :param catalog:
    :param filenames:
    :return:
    """
    return



def delete_ids_by_faiss(catalog: str, ids: List[int]):
    """
    根据索引id列表删除对应索引
    :param catalog:
    :param ids:
    :return:
    """
    # 根据文件名获得对应索引
    index = faiss.read_index(catalog)
    # 根据索引id删除对应索引
    remove_set = np.array(ids, dtype=np.int64)
    index.remove_ids(remove_set)
    # 落盘
    faiss.write_index(index, catalog)