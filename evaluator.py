class Evaluator:
    @staticmethod
    def score(prediction: str, actual_trend: str, actual_amplitude: str) -> int:
        """
        评分规则：
        1. 预测与实际方向相同（都上升/都下降）→ 80分
        2. 预测为持平，实际为小幅上升/下降；或预测为小幅上升/下降，实际为持平 → 40分
        3. 预测与实际方向相反（一升一降）→ 0分
        4. 实际趋势未知 → -1（无法评分）
        """
        if actual_trend == "未知":
            return -1

        # 方向相同（上升/下降匹配）
        if prediction == actual_trend:
            return 80

        # 处理持平与小幅波动的相似情况
        is_actual_small = "小幅" in actual_amplitude or "轻微" in actual_amplitude
        if (prediction == "持平" and is_actual_small) or \
                (actual_trend == "持平" and prediction in ["上升",
                                                           "下降"] and is_actual_small):
            return 40

        # 方向相反
        return 0