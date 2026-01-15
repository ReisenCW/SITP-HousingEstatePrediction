from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS
import re
import json

class EvaluatorAgent(BaseAgent):
    def __init__(self, config, model=None):
        super().__init__(config, model=model or config.FAST_MODEL)

    @staticmethod
    def calculate_score(prediction_phases: list, actual_phases: list) -> int:
        """根据解析出的阶段信息，用代码计算分数"""
        if not prediction_phases or not actual_phases:
            return -1
            
        scores = []
        max_length = max(len(prediction_phases), len(actual_phases))
        
        for i in range(max_length):
            pred_phase = prediction_phases[i] if i < len(prediction_phases) else prediction_phases[-1]
            actual_phase = actual_phases[i] if i < len(actual_phases) else actual_phases[-1]
            
            if (pred_phase["trend"] in ["上升"] and actual_phase["trend"] in ["下降"]) or \
               (pred_phase["trend"] in ["下降"] and actual_phase["trend"] in ["上升"]):
                scores.append(0)
                continue
                
            d = abs(pred_phase["magnitude"] - actual_phase["magnitude"])
            phase_score = max(0, round(100 - 25 * d))
            
            if (pred_phase["trend"] == "持平" and actual_phase["trend"] in ["上升", "下降"] and 
                actual_phase["magnitude"] > 3):
                phase_score = max(0, phase_score - 10)
                
            scores.append(phase_score)
            
        return round(sum(scores) / len(scores)) if scores else -1

    async def evaluate(self, content: str) -> list:
        """使用大模型解析趋势内容，提取阶段和幅度"""
        prompt = PROMPTS["evaluator_parse_trend"].format(content=content)
        try:
            response = self._call(prompt, system_prompt=SYSTEM_PROMPTS["evaluator"])
            json_str = re.search(r'\[.*]', response, re.DOTALL).group()
            return json.loads(json_str)
        except Exception as e:
            print(f"解析趋势失败: {e}")
            return []

    async def score(self, prediction: str, actual_trend: str) -> int:
        """通过LLM解析预测与实际的阶段和幅度，然后用代码计算分数（0-100）。"""
        prediction_phases = await self.evaluate(prediction)
        actual_phases = await self.evaluate(actual_trend)
        return self.calculate_score(prediction_phases, actual_phases)
