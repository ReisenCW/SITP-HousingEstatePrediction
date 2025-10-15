
from config import Config
from evaluator import Evaluator
from agent import HousePriceAgent
from dashscope import Generation


def test():
    prompt = f"""
                请联网查询黄浦区董家渡在2024上半年的实际房价趋势，输出格式：
                趋势：[上升/下降/持平]
                幅度描述：[如“大幅上升”“小幅下降”“基本持平”等]
                """
    response = Generation.call(
        model=Config.MODEL if hasattr(Config, 'MODEL') else 'qwen-long',
        prompt=prompt,
        enable_search=True,
        result_format='text'
    )
    print(response.output.text)

def test_normalize_cot():
    """本地测试 _normalize_cot 的解析逻辑（无需联网）。"""
    cfg = Config()
    agent = HousePriceAgent(cfg)
    sample_cot = (
        "1. 政策分析：放松限购（2024-06，市级，适用于部分城区）；预计短期内新增成交量+10%。\n"
        "2. 区域特性：近6月网签量=25500套（同比+30%），历史年化波动=±6%。\n"
        "3. 量价关系：网签量↑30%，成交均价环比+0.3%，说明以价换量为主。\n"
        "4. 时间边界：政策公布于预测周期前1月，预计影响滞后1-2月。\n"
        "5. 历史反思复用：参考2016年X市放松限购案例，价格仅小幅上升。\n"
        "6. 结论（预测结果）：小幅上升，约1%（置信度中等）。"
    )
    parsed = agent._normalize_cot(sample_cot)
    print("Parsed COT:")
    for k, v in parsed.items():
        print(f"{k}: {v}\n")


if __name__ == "__main__":
    test_normalize_cot()