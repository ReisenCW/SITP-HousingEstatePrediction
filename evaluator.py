class Evaluator:
    @staticmethod
    def score(prediction: str, actual_trend: str) -> int:
        """
        评分规则：
        1. 预测与实际方向相同（如“上升”/“下降”/“持平”）→ 80分
        2. 预测为持平，实际为小幅上升/下降；或预测为小幅上升/下降，实际为持平 → 40分
        3. 预测与实际方向相反（一升一降）→ 0分
        4. 实际趋势未知 → -1（无法评分）
        """
        def parse_trend(text):
            parts = text.split(',')
            if len(parts) == 2:
                amplitude, trend = parts[0].strip(), parts[1].strip()
            else:
                amplitude, trend = '', text.strip()
            return amplitude, trend

        pred_amp, pred_trend = parse_trend(prediction)
        act_amp, act_trend = parse_trend(actual_trend)

        if act_trend == "未知":
            return -1

        # 方向相同（上升/下降/持平匹配）
        if pred_trend == act_trend:
            return 80

        # 处理持平与小幅波动的相似情况
        is_actual_small = "小幅" in act_amp or "轻微" in act_amp or "基本" in act_amp
        is_pred_small = "小幅" in pred_amp or "轻微" in pred_amp or "基本" in pred_amp
        if (pred_trend == "持平" and is_actual_small) or \
           (act_trend == "持平" and is_pred_small):
            return 40

        # 方向相反 或 一个大幅上升/下降, 一个持平
        return 0