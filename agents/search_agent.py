import time
from agents.base_agent import BaseAgent
from llm_utils import call_llm
from prompts import PROMPTS, SYSTEM_PROMPTS

class SearchAgent(BaseAgent):
    def __init__(self, config, model=None):
        super().__init__(config, model=model or config.FAST_MODEL)
    
    async def search(self, region: str, time_range: str) -> str:
        use_prompt = PROMPTS["search_related_info"].format(
            region=region, 
            time_range=time_range
        )

        if getattr(self.config, 'DEBUG', False):
            print(f"[SearchAgent] 正在执行联网搜索...")
        
        search_start = time.time()
        raw_info = self._call(use_prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)
        search_duration = time.time() - search_start

        if getattr(self.config, 'DEBUG', False):
            print(f"[SearchAgent] 联网搜索完成，耗时: {search_duration:.2f}s")
        
        # 自动精简信息以提升后续推理效率
        if raw_info and len(raw_info) > 500:
            if getattr(self.config, 'DEBUG', False):
                print(f"[SearchAgent] 原始信息过长({len(raw_info)}字)，正在精简核心要点...")
            
            summary_start = time.time()
            summary_prompt = PROMPTS["summarize_info"].format(info=raw_info)
            # 使用 FAST_MODEL 快速精简，设置 max_tokens=800 强制简练
            summarized = call_llm(
                model=self.config.FAST_MODEL, 
                prompt=summary_prompt, 
                system_prompt="你是一个信息精炼助手",
                max_tokens=800
            )
            summary_duration = time.time() - summary_start

            if getattr(self.config, 'DEBUG', False):
                print(f"[SearchAgent] 精简耗时: {summary_duration:.2f}s")
            
            return summarized if summarized else raw_info
        return raw_info

    async def get_actual(self, region: str, time_range: str) -> str:
        prompt = PROMPTS["get_actual_trend"].format(
            region=region, 
            time_range=time_range
        )
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["search_expert"], search=True)
