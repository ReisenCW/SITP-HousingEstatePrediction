import os

api_key = os.getenv("DASHSCOPE_API_KEY")

class Config:
    DEBUG = True
    MAX_ITERATIONS = 3
    API_KEY = api_key
    top_k = 7
