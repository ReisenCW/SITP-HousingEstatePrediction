from dashscope import Generation
from prompts import PROMPTS
from config import Config
import json
import re

class Evaluator:
    # 修复提示词中的JSON示例转义问题
    PARSE_TREND_PROMPT = PROMPTS["evaluator_parse_trend"]
    
    @staticmethod
    def parse_trend(content: str) -> list:
        """使用大模型解析趋势内容，提取阶段和幅度"""
        prompt = Evaluator.PARSE_TREND_PROMPT.format(content=content)
        try:
            response = Generation.call(
                model=Config.MODEL,
                prompt=prompt,
                result_format="text"
            )
            # 提取JSON部分
            json_str = re.search(r'\[.*]', response.output.text, re.DOTALL).group()
            return json.loads(json_str)
        except Exception as e:
            print(f"解析趋势失败: {e}")
            return []
    
    @staticmethod
    def calculate_score(prediction_phases: list, actual_phases: list) -> int:
        """根据解析出的阶段信息，用代码计算分数"""
        if not prediction_phases or not actual_phases:
            return -1
            
        scores = []
        max_length = max(len(prediction_phases), len(actual_phases))
        
        for i in range(max_length):
            # 处理长度不匹配的情况，使用最后一个阶段进行比较
            pred_phase = prediction_phases[i] if i < len(prediction_phases) else prediction_phases[-1]
            actual_phase = actual_phases[i] if i < len(actual_phases) else actual_phases[-1]
            
            # 趋势完全相反直接得0分
            if (pred_phase["trend"] in ["上升"] and actual_phase["trend"] in ["下降"]) or \
               (pred_phase["trend"] in ["下降"] and actual_phase["trend"] in ["上升"]):
                scores.append(0)
                continue
                
            # 计算幅度差异
            d = abs(pred_phase["magnitude"] - actual_phase["magnitude"])
            phase_score = max(0, round(100 - 30 * d))
            
            # 处理明显方向性矛盾
            if (pred_phase["trend"] == "持平" and actual_phase["trend"] in ["上升", "下降"] and 
                actual_phase["magnitude"] > 3):
                phase_score = max(0, phase_score - 10)
                
            scores.append(phase_score)
            
        # 计算平均分数
        return round(sum(scores) / len(scores)) if scores else -1
    
    @staticmethod
    def score(prediction: str, actual_trend: str) -> int:
        """
        通过LLM解析预测与实际的阶段和幅度，然后用代码计算分数（0-100）。
        """
        # 解析预测和实际结果
        prediction_phases = Evaluator.parse_trend(prediction)
        actual_phases = Evaluator.parse_trend(actual_trend)
        
        # 计算分数
        return Evaluator.calculate_score(prediction_phases, actual_phases)