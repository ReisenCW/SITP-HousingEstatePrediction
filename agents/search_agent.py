from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class SearchAgent(BaseAgent):
    def __init__(self, config, model=None):
        super().__init__(config, model=model or config.WEB_MODEL)
    
    async def search(self, region: str, time_range: str, reflections: str = "") -> str:
        prompt = PROMPTS["search_related_info"].format(
            region=region, 
            time_range=time_range
        )

        # 调整 prompt 策略
        if reflections:
            adjust_prompt = f"请根据先前的反思结果{reflections}, 修改原先用于向LLM联网搜索影响房价的政策、新闻等信息的prompt, 调整搜索策略, 避免犯同样的错误。原prompt: {prompt}"
            adjusted = self._call(adjust_prompt, system_prompt=SYSTEM_PROMPTS["reflector"])
            use_prompt = adjusted if adjusted else prompt
        else:
            use_prompt = prompt

        return self._call(use_prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)

    async def get_actual(self, region: str, time_range: str) -> str:
        prompt = PROMPTS["get_actual_trend"].format(region=region, time_range=time_range)
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)
