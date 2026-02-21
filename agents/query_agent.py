import re
from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class QueryAgent(BaseAgent):
    def __init__(self, config, model=None):
        super().__init__(config, model=model or config.FAST_MODEL)

    async def parse(self, query: str) -> tuple:
        prompt = PROMPTS["parse_query"].format(query=query)
        result = self._call(prompt, system_prompt=SYSTEM_PROMPTS["query_parser"])
        
        if not result:
            raise RuntimeError("解析问题失败：模型未返回有效输出")
        
        try:
            # 使用正则表达式提取，支持全角和半角冒号，并忽略前后空白
            region_match = re.search(r"区域[:：]\s*(.*)", result)
            time_match = re.search(r"时间范围[:：]\s*(.*)", result)
            
            if not region_match or not time_match:
                raise IndexError("未能匹配区域或时间范围")
                
            region = region_match.group(1).strip().strip("[]").strip()
            time_range = time_match.group(1).strip().strip("[]").strip()
            
            return region, time_range
        except (IndexError, AttributeError):
            raise RuntimeError(f"解析结果格式不符合预期: {result}")
