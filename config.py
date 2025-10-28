import os

class Config:
    # API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 阿里云DashScope API密钥
    MODEL = "qwen3-max"  # 使用的模型名称
    # 迭代与搜索配置
    MAX_ITERATIONS = 1  # 预测时的最大搜索迭代次数
    MAX_RETRIES = 4  # 预测错误时的最大重试次数
    SEARCH_TIMEOUT = 30  # 搜索超时时间（秒）
    SCORE_THRESHOLD = 80  # 评分阈值，低于该分数则进行反思和重试
    # 是否进化
    ENABLE_EVOLUTION = True  # 是否启用进化
    # 日志与存储
    DEBUG = True  # 是否开启调试日志
    REFLECTION_HISTORY_PATH = "reflection_history.json"  # 反思记录路径
    COT_TRAJECTORY_PATH = "cot_trajectory.json"  # COT轨迹记录路径
    ANSWER_PATH = "./answer/"  # 最终答案存储路径
    # COT / reflection 控制选项：
    # 可选值 'ENABLE_COT', 'ENABLE_REFLECTION', 'ENABLE_BOTH', 'ENABLE_NONE'
    HISTORY_MODE = 'ENABLE_BOTH'

    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("请设置环境变量DASHSCOPE_API_KEY（从阿里云DashScope获取）")
        # 验证 HISTORY_MODE
        allowed = {'ENABLE_COT', 'ENABLE_REFLECTION', 'ENABLE_BOTH', 'ENABLE_NONE'}
        if getattr(cls, 'HISTORY_MODE', None) not in allowed:
            raise ValueError(f"无效的 HISTORY_MODE 配置: {cls.HISTORY_MODE}，应为 {allowed}")