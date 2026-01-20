from llm_utils import call_llm
from config import Config
import time

class BaseAgent:
    def __init__(self, config: Config, model: str = None, think: bool = False):
        self.config = config
        self.model = model or config.MODEL
        self.enable_thinking = think

    def _call(self, prompt: str, system_prompt: str = None, search: bool = False) -> str:
        start_time = time.time()
        result = call_llm(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            search=search
        )
        duration = time.time() - start_time
        if getattr(self.config, 'DEBUG', False):
            print(f"[{self.__class__.__name__}] LLM调用耗时: {duration:.2f}秒")
        return result
