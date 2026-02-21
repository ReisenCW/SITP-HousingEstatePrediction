from llm_utils import call_llm
from config import Config
import time
import sys

class BaseAgent:
    def __init__(self, config: Config, model: str = None, think: bool = False):
        self.config = config
        self.model = model or config.MODEL
        self.enable_thinking = think

    def _call(self, prompt: str, system_prompt: str = None, search: bool = False, stream: bool = False, max_tokens: int = None) -> str:
        start_time = time.time()
        result_or_gen = call_llm(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            search=search,
            stream=stream,
            max_tokens=max_tokens
        )
        
        if stream:
            print(f"[{self.__class__.__name__}] 开始推理...", flush=True)
            full_content = ""
            for chunk in result_or_gen:
                full_content += chunk
                print(chunk, end="", flush=True)
            print("\n")
            result = full_content
        else:
            result = result_or_gen

        duration = time.time() - start_time
        if getattr(self.config, 'DEBUG', False):
            print(f"[{self.__class__.__name__}] LLM调用耗时: {duration:.2f}秒")
        return result
