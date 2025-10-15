from dashscope import Generation
from prompts import PROMPTS

class Evaluator:
    LLM_SCORE_RULE = PROMPTS["evaluator_llm_score_rule"]
    @staticmethod
    def score(prediction: str, actual_trend: str) -> int:
        """
        通过LLM判断预测与实际的匹配程度，输出分数（0-100）。
        """
        prompt = PROMPTS["evaluator_score"].format(
            LLM_SCORE_RULE=Evaluator.LLM_SCORE_RULE,
            prediction=prediction,
            actual_trend=actual_trend
        )
        response = Generation.call(
            model="qwen-plus",
            prompt=prompt,
            result_format="text"
        )
        score_str = response.output.text.strip().split()[0]
        try:
            score = int(score_str)
        except Exception:
            score = -1
        return score
        