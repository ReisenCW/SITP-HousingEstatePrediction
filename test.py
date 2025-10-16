import asyncio
from config import Config
from agent import HousePriceAgent
from evaluator import Evaluator

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

async def main():
    # 初始化配置和智能体
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误：{e}")
        return

    agent = HousePriceAgent(config)

    q_regions = ["豫园", "徐汇滨江", "董家渡", "新天地"]
    q_periods = [
        "2024年上半年",
        "2024年下半年",
        "2025年上半年"
    ]

    for region in q_regions:
        for j in range(len(q_periods)):
            user_query = f"{region}{q_periods[j]}房价走势如何"
            print(f"\n===== 开始处理: {user_query} =====\n")
            region, time_range = await agent.parse_query(user_query)

            info = await agent.search_related_info(region, time_range)
            prediction = await agent.predict_trend(region, time_range,
                                                    info)
            actual_result = await agent.get_actual_trend(region, time_range)
            score = Evaluator.score(prediction, actual_result)
            reflection = await agent.generate_reflection(
                user_query, prediction, actual_result, score, info
            )
            print(f"预测趋势：{prediction}\n实际趋势：{actual_result}\n评分：{score}\n")
            agent.save_answer(user_query, prediction, actual_result, score)
                
if __name__ == "__main__":
    asyncio.run(main())