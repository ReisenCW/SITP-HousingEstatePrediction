import os

class Config:
    # API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 阿里云DashScope API密钥
    MODEL = "qwen-plus"  # 使用的模型名称
    # 迭代与搜索配置
    MAX_ITERATIONS = 3  # 预测时的最大搜索迭代次数
    SEARCH_TIMEOUT = 30  # 搜索超时时间（秒）
    # 日志与存储
    DEBUG = True  # 是否开启调试日志
    REFLECTION_HISTORY_PATH = "reflection_history.md"  # 反思记录路径

    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("请设置环境变量DASHSCOPE_API_KEY（从阿里云DashScope获取）")