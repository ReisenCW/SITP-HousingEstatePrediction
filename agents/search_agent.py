from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class SearchAgent(BaseAgent):
    def __init__(self, config, model=None):
        super().__init__(config, model=model or config.FAST_MODEL)
    
    async def search(self, region: str, time_range: str, reflections: str = "") -> str:
        prompt = PROMPTS["search_related_info"].format(
            region=region, 
            time_range=time_range
        )

        # 调整 prompt 策略
        if reflections:
            if getattr(self.config, 'DEBUG', False):
                print(f"[SearchAgent] 正在根据反思记录优化搜索策略...")
            
            adjust_prompt = PROMPTS["adjust_search_prompt"].format(
                reflections=reflections,
                original_prompt=prompt,
                region=region,
                time_range=time_range
            )

            adjusted = self._call(adjust_prompt, system_prompt=SYSTEM_PROMPTS["reflector"])
            
            # 清理可能存在的 Markdown 代码块包裹和空白
            if adjusted:
                adjusted = adjusted.strip()
                # 移除 ```prompt, ```text, ``` 等包裹
                if adjusted.startswith("```"):
                    lines = adjusted.splitlines()
                    if len(lines) > 1 and lines[0].startswith("```"):
                        # 找到结尾的 ```
                        end_idx = -1
                        for i in range(len(lines)-1, 0, -1):
                            if lines[i].strip() == "```":
                                end_idx = i
                                break
                        if end_idx != -1:
                            adjusted = "\n".join(lines[1:end_idx]).strip()
                        else:
                            adjusted = "\n".join(lines[1:]).strip()
            
            use_prompt = adjusted if adjusted else prompt
            if getattr(self.config, 'DEBUG', False):
                # print(f"[OPTIMIZED PROMPT CLIP] {use_prompt[:100]}...")
                print(f"[OPTIMIZED PROMPT CLIP] {use_prompt}")
        else:
            use_prompt = prompt

        if getattr(self.config, 'DEBUG', False):
            print(f"[SearchAgent] 正在执行联网搜索...")
        return self._call(use_prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)

    async def get_actual(self, region: str, time_range: str) -> str:
        prompt = PROMPTS["get_actual_trend"].format(region=region, time_range=time_range)
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)
