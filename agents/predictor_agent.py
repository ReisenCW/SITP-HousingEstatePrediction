from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class PredictorAgent(BaseAgent):
    async def predict(self, region: str, time_range: str, info: str, 
                      reflection_history: str = "", recent_cot: str = "") -> str:
        prompt = PROMPTS["predict_trend"].format(
            region=region,
            time_range=time_range,
            info=info,
            reflection_history=f"历史反思记录：\n{reflection_history}" if reflection_history else "",
            recent_cot=f"最近3条思维链（COT）记录：\n{recent_cot}" if recent_cot else ""
        )
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["trend_predictor"])
