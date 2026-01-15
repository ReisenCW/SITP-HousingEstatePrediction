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
            # 提取“：”后的文字，并移除两端的空白字符，再移除两端可能的方括号及其内部/外部空白
            region = [line.split("：")[1].strip().strip("[]").strip() for line in result.split("\n") if "区域：" in line][0]
            time_range = [line.split("：")[1].strip().strip("[]").strip() for line in result.split("\n") if "时间范围：" in line][0]
            return region, time_range
        except (IndexError, AttributeError):
            raise RuntimeError(f"解析结果格式不符合预期: {result}")
