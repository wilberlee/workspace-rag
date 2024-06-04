#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/5/17 09:30
# @Author: lbl
import os
import logging
from datetime import datetime


def setup_logger():
    # 日志文件命名
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_filename = "output_" + datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(log_folder, log_filename)
    # 检查日志文件夹是否存在，如果不存在则创建
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建日志记录器
    # logger = logging.getLogger('NLP')
    logger = logging.getLogger(__name__)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 设置处理器的格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建全局的日志记录器
logger = setup_logger()
