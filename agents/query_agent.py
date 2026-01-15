from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class QueryAgent(BaseAgent):
    async def parse(self, query: str) -> tuple:
        prompt = PROMPTS["parse_query"].format(query=query)
        result = self._call(prompt, system_prompt=SYSTEM_PROMPTS["query_parser"])
        
        if not result:
            raise RuntimeError("解析问题失败：模型未返回有效输出")
        
        try:
            region = [line.split("：")[1].strip("[]") for line in result.split("\n") if "区域：" in line][0]
            time_range = [line.split("：")[1].strip("[]") for line in result.split("\n") if "时间范围：" in line][0]
            return region, time_range
        except (IndexError, AttributeError):
            raise RuntimeError(f"解析结果格式不符合预期: {result}")
