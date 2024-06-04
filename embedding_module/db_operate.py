# @Description : 提供对向量库操作的接口，包括插入数据、删除数据、删除集合、查询
import os
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from configs.model_config import VS_ROOT_PATH
from utils.logger import logger
from configs.db_config import OUTPUT_FIELDS
import milvus_api, faiss_api


def init_milvus(host: str, port: str):
    """
    milvus初始化
    :param host: ip
    :param port: port
    :return:
    """
    try:
        logger.info("host is {}, port is {}".format(host, port))
        return milvus_api.init_milvus(host, port)
    except Exception as e:
        logger.error("init_milvus:{}".format(e), exc_info=True)

def get_milvus_collection_nums(foleder_name: str):
    """
    获取当前collection下的数据量
    :param foleder_name:
    :return:
    """
    return milvus_api.get_collection_nums(foleder_name)

def insert_data_by_milvus(folder_name: str,
                          filename: str or List[str],
                          docs: List[str],
                          embeddings: Embeddings,
                          index_name: str = "index"):
    """
    insert_data_by_milvus
    :param folder_name: 知识库名称
    :param filename: 文件名称
    :param docs: 文本集
    :param embeddings:向量化模型
    :param index_name: 索引名称
    :return:
    """
    try:
        # logger.info("folder_name:{}, filename:{}, len_docs:{}, index_name:{}".format(folder_name, filename, len(docs), index_name))
        return milvus_api.insert_data_into_milvus(folder_name, filename, docs, embeddings, index_name)
    except Exception as e:
        logger.error("insert_data_by_milvus:{}".format(e), exc_info=True)


# def insert_data_by_milvus(folder_name: str,
#                           filenames: List[str],
#                           docs: List[str],
#                           embeddings: Embeddings,
#                           index_name: str = "index"):
#     """
#     insert_data_by_milvus
#     :param folder_name: 知识库名称
#     :param filename: 文件名臣
#     :param docs: 文本集
#     :param embeddings:向量化模型
#     :param index_name: 索引名称
#     :return:
#     """
#     try:
#         logger.info("folder_name:{}, filenames:{}, len_docs:{}, index_name:{}".format(folder_name, filenames, len(docs), index_name))
#         return milvus_api.insert_data_into_milvus(folder_name, filenames, docs, embeddings, index_name)
#     except Exception as e:
#         logger.error("insert_data_by_milvus:{}".format(e), exc_info=True)

def insert_data_by_faiss(vs_path: str,
                         embeddings: Embeddings,
                         docs: List[Document],
                         catalogs: str = "",
                         index_name: str = "index"):
    """
    insert_data_by_faiss
    :param vs_path: 知识库路径
    :param embeddings: 向量化模型
    :param docs: 文本集
    :param catalogs: 知识库名称
    :param index_name: 索引名称
    :return:
    """
    try:
        logger.info("vs_path:{}, len_docs:{}, catalogs:{}, index_name:{}".format(vs_path, len(docs), catalogs, index_name))
        return faiss_api.insert_data_by_faiss(vs_path, embeddings, docs, catalogs, index_name)
    except Exception as e:
        logger.error("insert_data_by_faiss:{}".format(e), exc_info=True)



def delete_vector_store_by_milvus(folder_name: str):
    """
    delete_vector_store_by_milvus
    :param folder_name: 知识库名称
    :return:
    """
    try:
        logger.info("folder_name:{}".format(folder_name))
        return milvus_api.delete_vector_store(folder_name)
    except Exception as e:
        logger.error("delete_vector_store_by_milvus:{}".format(e), exc_info=True)

def delete_data_by_milvus(folder_name: str, filename: str or List[str]):
    """
    delete_data_by_milvus
    :param folder_name: 知识库名称
    :param filename: 文件名，可以是多个
    :return:
    """
    try:
        logger.info("folder_name:{}, filename:{}".format(folder_name, filename))
        ids = []
        if type(filename) is str:
            results = milvus_api.query_by_filename(folder_name, filename)
            for result in results:
                ids.append(result["id"])
            return milvus_api.delete_vectors_with_ids(folder_name, ids)
        else:
            for each in filename:
                results = milvus_api.query_by_filename(folder_name, each)
                for result in results:
                    ids.append(result["id"])
            return milvus_api.delete_vectors_with_ids(folder_name, ids)
    except Exception as e:
        logger.error("delete_data_by_milvus:{}".format(e), exc_info=True)




def delete_vector_store_by_faiss(catalogs: List):
    logger.info("catalogs:{}".format(catalogs))
    try:
        # 传进来的是库名列表["知识库1", "知识库2"]
        # catalogs_lst = request.json['catalogs']
        for catalog in catalogs:
            folder_path = os.path.join(VS_ROOT_PATH, catalog)
            if not folder_path:
                logger.error("Folder path is missing,{}".format(folder_path))
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(folder_path)
                logger.info('File {} has been deleted.'.format(folder_path))
            else:
                logger.error('The file {} does not exist.'.format(folder_path))
    except Exception as e:
        logger.error('Error while deleting vector library folder:{}'.format(e))




def delete_data_by_faiss(catalog: str, filenames: str or  List[str]):
    logger.info("catalog:{}, filenames:{}".format(catalog, filenames))
    try:
        if type(filenames) is str:
    #         单文件删除
            faiss_api.delete_data_by_faiss(catalog,filenames)

        else:
            for filename in filenames:
                faiss_api.delete_data_by_faiss(catalog, filename)
    #         多文件删除
    except Exception as e:
        logger.error("delete_data_by_faiss:{}".format(e), exc_info=True)
    return



def search_query_by_milvus(folder_name: str,
                           embeddings: Embeddings,
                           query: str or List[str] or List[float] or List[List[float]],
                           topk: int = 4):
    """
    search_query_by_milvus,query可以是vector、vectors、text、texts
    :param folder_name: 知识库名称
    :param embeddings: 向量化模型
    :param query: query
    :param topk: 返回数量，默认4ge
    :return: [{
    "score":int,
    "entity":entity
    }]
    List[Dict[str, Union[int, Entity]]]
    """
    try:
        logger.info("folder_name:{}, query:{}, topk:{}".format(folder_name, query, topk))
        result_lists = []
        if type(query) is str:
            vectors = embeddings.embed_documents([query])
            # result = milvus_api.search_by_text(folder_name, embeddings, query, topk)
        elif type(query) is list and type(query[0]) is str:
            vectors = embeddings.embed_documents(query)
            # result = milvus_api.search_by_texts(folder_name, embeddings, query, topk)
        elif type(query) is list and type(query[0]) is float:
            vectors = [query]
            # result = milvus_api.search_by_vector(folder_name, query, topk)
        else:
            vectors = query
            # result = milvus_api.search_by_vectors(folder_name, query, topk)
        result = milvus_api.search_by_vectors(folder_name, vectors, topk, output_fields=OUTPUT_FIELDS)
        dim = len(vectors[0])
        for hits in result:
            for hit in hits:
                hit_dict = {"score": hit.distance, "entity": hit.entity}
                result_lists.append(hit_dict)
        return result_lists
    except Exception as e:
        logger.error("search_query_by_milvus:{}".format(e), exc_info=True)


def search_query_with_score_by_faiss(vs_path: str,
                                     embeddings: Embeddings,
                                     index_name: str,
                                     query: str or List[str] or List[float] or List[List[float]],
                                     similarity_search_function,
                                     topk: int = 4):
    """
    search_query_with_score_by_faiss,query可以是vector、vectors、text、texts
    :param vs_path: 知识库路径
    :param embeddings: 向量化模型
    :param index_name: 索引名称
    :param query: query
    :param similarity_search_function: 重定义函数
    :param topk: 返回个数，默认四个
    :return: List[Tuple[Document, float]]
    """
    try:
        logger.info("vs_path:{}, index_name:{}, query:{}, topk:{}".format(vs_path, index_name, query, topk))
        if type(query) is str:
            return faiss_api.search_with_score_by_text(vs_path, embeddings, index_name, query, similarity_search_function, topk)
        elif type(query) is list and type(query[0]) is str:
            return faiss_api.search_with_score_by_texts(vs_path, embeddings, index_name, query, similarity_search_function, topk)
        elif type(query) is list and type(query[0]) is float:
            return faiss_api.search_with_score_by_vector(vs_path, embeddings, index_name, query, topk)
        else:
            return faiss_api.search_with_score_by_vectors(vs_path, embeddings, index_name, query, topk)
    except Exception as e:
        logger.error("search_query_with_score_by_faiss:{}".format(e), exc_info=True)

def query_by_id_by_milvus(folder_path: str, data_id: int):
    try:
        logger.info("folder_path:{}, data_id:{}".format(folder_path, data_id))
        # print(folder_path, data_id)
        return milvus_api.query_by_id(folder_path, data_id, output_fields=OUTPUT_FIELDS)
    except Exception as e:
        logger.error("query_by_id_by_milvus:{}".format(e), exc_info=True)
        return None


def query_by_ids_by_milvus(folder_path: str, data_ids: List[int]):
    try:
        # logger.info("folder_path:{}, data_id:{}".format(folder_path, data_ids))
        # print(folder_path, data_ids)
        return milvus_api.query_by_ids(folder_path, data_ids, output_fields=OUTPUT_FIELDS)
    except Exception as e:
        logger.error("query_by_id_by_milvus:{}".format(e), exc_info=True)
        return None