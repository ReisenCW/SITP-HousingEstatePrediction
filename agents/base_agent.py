from llm_utils import call_llm
from config import Config

class BaseAgent:
    def __init__(self, config: Config, model: str = None):
        self.config = config
        self.model = model or config.MODEL

    def _call(self, prompt: str, system_prompt: str = None, search: bool = False) -> str:
        return call_llm(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            search=search
        )
