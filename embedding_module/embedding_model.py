#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/5/18 16:52

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pypinyin import lazy_pinyin

from configs.model_config import *
from db_operate import (
    insert_data_by_faiss,
    insert_data_by_milvus,
    search_query_with_score_by_faiss,
    search_query_by_milvus,
)
from utils.logger import logger
from utils import torch_gc
from chains.local_doc_common import (
    similarity_search_with_score_by_milvus,
    similarity_search_with_score_by_faiss,
)


class VectorModel:
    def __init__(self):
        self.embeddings = None

    def init_vector_model(self, embedding_model, embedding_device=EMBEDDING_DEVICE):
        """
        加载向量化模型
        :param embedding_model: 向量化模型名称，对应embedding_model_dict的key
        :param embedding_device: GPU/CPU
        :return:
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model],
            model_kwargs={"device": embedding_device},
        )

    def init_vector_base_file(
        self, vs_path, docs, catalogs, db_type, index_name: str = "index"
    ):
        """
        文档存入faiss/milvus
        :param vs_path:
        :param docs:
        :param catalogs:
        :param db_type:
        :param index_name:
        :return:
        """
        try:
            if db_type == "faiss":
                insert_data_by_faiss(
                    vs_path, embeddings=self.embeddings, docs=docs, catalogs=catalogs
                )

            elif db_type == "milvus":
                # 从docs的List[Document]结构里取出page_content和filename
                texts = [doc.page_content for doc in docs]
                filenames = [doc.metadata["source"] for doc in docs]
                assert (
                    len(texts) == len(filenames) and texts
                ), "列表texts和列表filenames长度不一致"
                # 集合名只能是字母、数字、下划线
                catalogs = "".join(lazy_pinyin(catalogs))
                insert_data_by_milvus(
                    folder_name=catalogs,
                    filename=filenames,
                    docs=texts,
                    embeddings=self.embeddings,
                    index_name=index_name,
                )
        except Exception as e:
            logger.error("文档存储到{}失败：{}".format(db_type, e))

    def get_vector_base_query(
        self,
        query,
        vs_catalogs,
        db_type,
        top_k=VECTOR_SEARCH_TOP_K,
        index_name: str = "index",
    ):
        """
        定义查询向量方法
        :param query:
        :param vs_catalogs:
        :param db_type:
        :param top_k:
        :param index_name:
        :return:
        """
        related_docs = []
        if db_type == "faiss":
            for vs_catalog in vs_catalogs:
                # 从faiss向量库中取
                vs_path = os.path.join(VS_ROOT_PATH, vs_catalog)
                related_docs_with_score = search_query_with_score_by_faiss(
                    vs_path,
                    self.embeddings,
                    index_name,
                    query,
                    similarity_search_function=similarity_search_with_score_by_faiss,
                    topk=top_k,
                )
                related_docs += related_docs_with_score
        elif db_type == "milvus":
            for vs_catalog in vs_catalogs:
                catalog = "".join(lazy_pinyin(vs_catalog))
                result_lists = search_query_by_milvus(
                    catalog, self.embeddings, query, top_k
                )
                if result_lists:
                    scores = [data["score"] for data in result_lists]
                    indices = [data["entity"].id for data in result_lists]
                    related_docs_with_score = similarity_search_with_score_by_milvus(
                        catalog, scores, indices
                    )
                    related_docs += related_docs_with_score
                else:
                    related_docs += []
        torch_gc()
        return related_docs
