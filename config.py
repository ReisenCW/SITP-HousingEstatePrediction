import os

api_key = os.getenv("DASHSCOPE_API_KEY")

class Config:
    DEBUG = True
    MAX_ITERATIONS = 2
    API_KEY = api_key
    PERSIST_DIRECTORY = "./local_search_db"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    LOCAL_SEARCH_TOP_K = 5  # 本地向量库返回top k条
    WEB_SEARCH_TOP_K = 15    # 网络检索返回top k条