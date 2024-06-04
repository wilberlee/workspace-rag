"""
@Project ：ai-langchain-chatglm
@File    ：__init__.py
@Author  ：DX
@Date    ：2023-05-19 21:13
@Description : 公共函数
"""
import torch


def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print(
                "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


# 将文件字节数转为MB或KB
def convert_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return "{:.1f}MB".format(size_bytes / (1024 * 1024))
    elif size_bytes >= 1024:
        return "{:.1f}KB".format(size_bytes / 1024)
    else:
        return "{}B".format(size_bytes)
