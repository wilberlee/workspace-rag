# -*- coding: utf-8 -*-
# @Time: 2024/5/29 9:56
# @File: main_api.py
# Desc:
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from embedding_module.model_config import *
import json
import io
import csv

app = FastAPI()


embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict["text2vec"],
            model_kwargs={"device": EMBEDDING_DEVICE},
        )

@app.post("/api/string_similarity/")
async def calculate_similarity_to_list(sentence, sentence_list):
    embedding = embeddings(sentence)
    pass

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()  # 读取文件内容
    file_extension = file.filename.split('.')[-1]  # 获取文件扩展名

    if file_extension == 'txt':
        return parse_txt(file_content)
    elif file_extension == 'json':
        return parse_json(file_content)
    elif file_extension == 'jsonl':
        return parse_jsonl(file_content)
    elif file_extension == 'csv':
        return parse_csv(file_content)
    else:
        return {"error": "Unsupported file type"}


def parse_txt(file_content: bytes):
    model_name = "chatglm3-6b"
    parser = process_txt_file(file_content, model_name)
    return parser


def parse_json(file_content: bytes):
    data = json.loads(file_content)
    return {"data": data}


def parse_jsonl(file_content: bytes):
    lines = file_content.decode('utf-8').splitlines()
    data = [json.loads(line) for line in lines]
    return {"data": data}


def parse_csv(file_content: bytes):
    file_content = io.StringIO(file_content.decode('utf-8'))
    reader = csv.DictReader(file_content)
    header = reader.fieldnames
    prompt_keys = header[:-1]
    response_key = header[-1]
    next(reader)  # 跳过第一行表头
    data = [row for row in reader]
    jsonl_data = []
    for item in data:
        prompt = " ".join(item[key] for key in prompt_keys)
        response = [[item[response_key]]]
        jsonl_item = [{"prompt": prompt, "response": response}]
        jsonl_data.append(jsonl_item)

    return jsonl_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
