# import asyncio
# from agent import Agent
# from config import Config
#
# async def run_all():
#     agent = Agent(Config())
#     place = "上海黄浦区新天地"
#     queries = [
#         "2023年下半年" + place + "房价具体走势如何？",
#         "2024年上半年" + place + "房价具体走势如何？",
#         "2024年下半年" + place + "房价具体走势如何？",
#         "2025年上半年" + place + "房价具体走势如何？"
#     ]
#     with open('answer.md', 'a', encoding='utf-8') as f:
#         for q in queries:
#             print(f"\n===== 预测: {q} =====")
#             result = await agent.llm_with_iteration(q)
#             f.write('\n' + '='*40 + f"\n* 预测: {q}\n* 结果:\n{result}\n")
#             print(result)
#
# if __name__ == "__main__":
#     asyncio.run(run_all())

from openai import OpenAI
from config import Config


def test():
    client = OpenAI(
        api_key=Config.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    prompt = "2023Q2-Q3徐汇滨江房价变化情况是什么？上升/下降 + 百分比"
    response = client.chat.completions.create(
        model="qwen-long",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_search": True}
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test()