from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class PredictorAgent(BaseAgent):
    def __init__(self, config, model=None, think=True):
        super().__init__(config, model=model or config.MODEL, think=think)
    
    async def predict(self, region: str, time_range: str, info: str, 
                      reflection_history: str = "") -> str:
        prompt = PROMPTS["predict_trend"].format(
            region=region,
            time_range=time_range,
            info=info,
            reflection_history=f"历史反思记录：\n{reflection_history}" if reflection_history else ""
        )
        # 设置 max_tokens 为 1500：
        # 1. 1500 token 约可容纳 1000-1200 个汉字，足以覆盖详细的 5-6 项维度分析。
        # 2. 限制过度生成有助于减少推理耗时，避免模型输出冗余的“车轮话”。
        # 3. 确保预测结果（通常在末尾）不会因截断而丢失。
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["trend_predictor"], stream=True, max_tokens=1500)
