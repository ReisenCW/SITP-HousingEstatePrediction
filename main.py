import asyncio
from config import Config
from agent import HousePriceAgent
from evaluator import Evaluator


async def main():
    # 初始化配置和智能体
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"配置错误：{e}")
        return

    agent = HousePriceAgent(config)

    # 获取用户输入
    user_query = input(
        "请输入房价预测问题（例如：2024年上海浦东新区房价走势如何？）：")
    print("\n===== 开始处理 =====")

    try:
        # 1. 解析问题（提取区域和时间范围）
        region, time_range = await agent.parse_query(user_query)
        print(f"解析结果：区域={region}，时间范围={time_range}")

        # 2. 联网搜索相关信息（多轮迭代补充）
        print("\n===== 联网搜索信息 =====")
        all_info = []
        for i in range(config.MAX_ITERATIONS):
            print(f"第{i + 1}轮搜索...")
            info = await agent.search_related_info(region, time_range)
            all_info.append(info)
        combined_info = "\n".join(all_info)

        # 3. 生成预测
        print("\n===== 生成预测 =====")
        prediction = await agent.predict_trend(region, time_range,
                                               combined_info)
        print(f"预测趋势：{prediction}")

        # 4. 获取实际趋势
        print("\n===== 获取实际趋势 =====")
        actual_trend, actual_amplitude = await agent.get_actual_trend(region,
                                                                      time_range)
        print(f"实际趋势：{actual_trend}（{actual_amplitude}）")

        # 5. 评估打分
        print("\n===== 评估结果 =====")
        evaluator = Evaluator()
        score = evaluator.score(prediction, actual_trend, actual_amplitude)
        if score == -1:
            print("无法评分（实际趋势数据未找到）")
        else:
            print(f"预测评分：{score}分")

        # 6. 生成反思（若可评分）
        if score != -1:
            print("\n===== 反思分析 =====")
            reflection = await agent.generate_reflection(
                user_query, prediction, actual_trend, score, combined_info
            )
            print(reflection)

    except Exception as e:
        print(f"处理失败：{e}")


if __name__ == "__main__":
    asyncio.run(main())