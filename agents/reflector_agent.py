from agents.base_agent import BaseAgent
from prompts import PROMPTS, SYSTEM_PROMPTS

class ReflectorAgent(BaseAgent):
    def __init__(self, config, model=None, think=True):
        super().__init__(config, model=model or config.MODEL, think=think)

    async def reflect(self, query: str, prediction: str, actual: str, 
                      score: int, info: str, history_reminder: str,
                      current_trajectory: str, 
                      persistent_memory: str) -> str:
        
        prompt_key = "generate_reflection_success" if score >= self.config.SCORE_THRESHOLD else "generate_reflection_failure"
        
        prompt = PROMPTS[prompt_key].format(
            score=score,
            history_reminder=history_reminder,
            query=query,
            current_trajectory=current_trajectory,
            persistent_memory=persistent_memory,
            prediction=prediction,
            actual=actual,
            info=info
        )
        
        return self._call(prompt, system_prompt=SYSTEM_PROMPTS["reflector"])
