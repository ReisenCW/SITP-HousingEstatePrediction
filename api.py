import asyncio
import re
from typing import Optional, Tuple
from agent import HousePriceAgent
from agents.evaluator_agent import EvaluatorAgent
from config import Config

def get_backtest_time_range(target_time: str) -> str:
    """
    根据目标时间（如 2025年上半年）推导出其前一个半年度（如 2024年下半年）。
    用于冷启动时的经验获取。
    """
    match = re.search(r"(\d{4})年(上|下)半年", target_time)
    if not match:
        # 默认回退到 2024年下半年
        return "2024年下半年"
    
    year = int(match.group(1))
    period = match.group(2)
    
    if period == "上":
        return f"{year-1}年下半年"
    else:
        return f"{year}年上半年"

async def perform_evolution_cycle(agent: HousePriceAgent, 
                                 region: str, 
                                 time_range: str, 
                                 max_retries: int, 
                                 evolve: bool = True,
                                 debug: bool = False) -> str:
    """执行单个时间范围的预测。如果 evolve=True，则包含评估+反思循环。"""
    retries = 0
    last_pred = "预测失败"
    
    while retries <= max_retries:
        # 获取最相似的记忆（反思）
        reflections = await agent.get_relevant_reflections(region)
        
        # 预测
        pred, info = await agent.predict(region, time_range, reflections)
        last_pred = pred
        
        if debug:
            print(f"[cycle] Prediction for {region} {time_range}: {pred}")

        # 如果不执行进化（回测流程），或者配置关闭了进化，则直接保存结果并退出
        if not evolve or not agent.config.ENABLE_EVOLUTION:
            agent.save_answer(f"{region} {time_range}", pred, "未获取实际趋势", -1)
            break

        actual = await agent.get_actual_trend(region, time_range)
        score = await agent.evaluator_agent.score(pred, actual)

        if debug:
            print(f"[cycle] Actual: {actual} | Score: {score}")

        _ = await agent.generate_reflection(f"{region} {time_range}", pred, actual, score, info)

        if score >= agent.config.SCORE_THRESHOLD or retries >= max_retries:
            agent.save_answer(f"{region} {time_range}", pred, actual, score)
            break

        retries += 1
        if debug:
            print(f"[cycle] Retry {retries}/{max_retries}...")
            
    return last_pred


async def predict_region_async(query: str,
                               max_retries: Optional[int] = None,
                               enable_evolution: bool = False,
                               debug: bool = False) -> str:
    """以异步方式运行所需的工作流。只需传入一段预测请求描述。"""
    config = Config()
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    agent = HousePriceAgent(config)
    
    # 1. 解析查询获取区域和目标时间
    region, target_time = await agent.parse_query(query)
    if debug:
        print(f"[api] Parsed Request: Region={region}, TargetTime={target_time}")

    # 2. 冷/热启动判断
    has_exp = await agent.check_experience(region)
    if not has_exp:
        backtest_range = get_backtest_time_range(target_time)
        if debug:
            print(f"[api] Cold Start detected for {region}. Backtesting {backtest_range} to gain experience...")
        # 冷启动：必须执行回测流程 (evolve=True) 以获取记忆
        await perform_evolution_cycle(agent, region, backtest_range, max_retries=1, evolve=True, debug=debug)
    else:
        if debug:
            print(f"[api] Hot Start for {region}. Using existing memory.")

    # 3. 执行核心目标预测 (是否进化由参数 enable_evolution 决定)
    prediction = await perform_evolution_cycle(agent, region, target_time, max_retries, evolve=enable_evolution, debug=debug)
    
    return prediction


def predict_region(query: str,
                   max_retries: Optional[int] = None,
                   enable_evolution: bool = False,
                   debug: bool = False) -> str:
    """predict_region_async 的同步包装器。"""
    return asyncio.run(predict_region_async(query, max_retries, enable_evolution, debug))

if __name__ == "__main__":
    # 只需输入你想预测的目标和时间
    test_query = "2025年下半年杨浦区五角场房价走势如何"
    
    # 默认 enable_evolution=False，仅获取预测结果
    result = predict_region(test_query, enable_evolution=False, debug=True)
    print("\n" + "="*50)
    print(f"最终预测结果 ({test_query}):")
    print(result)
    print("="*50)
