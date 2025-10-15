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

    user_query = "2024年下半年豫园房价走势如何"
    print("\n===== 开始处理 =====")

    def debug_log(msg):
        with open("log.md", "a", encoding="utf-8") as logf:
            logf.write(str(msg) + "\n")

    if config.DEBUG:
        with open("log.md", "a", encoding="utf-8") as logf:
            logf.write("\n===========================================\n"
                      f"预测: {user_query} ")

    try:
        retries = 0
        while retries <= config.MAX_RETRIES:
            # 1. 解析问题（提取区域和时间范围）
            region, time_range = await agent.parse_query(user_query)

            print(f"解析结果：区域={region}，时间范围={time_range}")
            if config.DEBUG:
                debug_log(f"[解析] 区域: {region}, 时间范围: {time_range}")

            # 2. 联网搜索相关信息（多轮迭代补充）
            print("\n===== 联网搜索信息 =====")
            all_info = []
            for i in range(config.MAX_ITERATIONS):
                print(f"第{i + 1}轮搜索...")
                info = await agent.search_related_info(region, time_range)
                all_info.append(info)
                if config.DEBUG:
                    debug_log(f"[搜索第{i+1}轮] {info}")
            combined_info = "\n".join(all_info)
            if config.DEBUG:
                debug_log(f"[全部搜索信息]\n{combined_info}")

            # 3. 生成预测
            print("\n===== 生成预测 =====")
            prediction = await agent.predict_trend(region, time_range,
                                                   combined_info)
            print(f"预测趋势：{prediction}")
            if config.DEBUG:
                debug_log(f"[预测趋势] {prediction}")

            if config.ENABLE_EVOLUTION:
                # 4. 获取实际趋势
                print("\n===== 获取实际趋势 =====")
                actual_result = await agent.get_actual_trend(region, time_range)

                print(f"实际趋势：{actual_result}")
                if config.DEBUG:
                    debug_log(f"[实际趋势] {actual_result}")

                # 5. 评估打分
                print("\n===== 评估结果 =====")
                score = Evaluator.score(prediction, actual_result)

                print(f"预测评分：{score}分")
                if config.DEBUG:
                    debug_log(f"[评估结果] 预测评分：{score}分")

                # 6. 生成反思（若非满分）
                print("\n===== 反思分析 =====")
                reflection = await agent.generate_reflection(
                    user_query, prediction, actual_result, score, combined_info
                )
                print(reflection)
                if config.DEBUG:
                    debug_log(f"[反思]\n{reflection}")
                if score >= config.SCORE_THRESHOLD:
                    agent.save_answer(user_query, prediction, actual_result, score)
                    break
                # 分数低于阈值则重试
                retries += 1
                if retries > config.MAX_RETRIES:
                    print(f"已达最大重试次数({config.MAX_RETRIES})，流程终止。")
                    agent.save_answer(user_query, prediction, actual_result, score)
                    break
                print(f"\n===== 重新预测（第{retries}次） =====")
                continue
            else:
                # 未启用进化则只跑一次
                agent.save_answer(user_query, prediction, "未获取实际趋势", None)
                break
    except Exception as e:
        print(f"处理失败：{e}")

if __name__ == "__main__":
    asyncio.run(main())