import os

def mkdirs(path):
    """创建目录（如果不存在的话）"""
    if not os.path.exists(path):
        os.makedirs(path)
