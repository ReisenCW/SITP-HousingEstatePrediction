import requests
from openai import OpenAI
import os
import sys

DEBUG = False

# 配置API密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
search_api_key = "f97a60574d1ce7566fcf06b15c8b07bf9eeebf21d94e9b2480bc96acccadf95b"
MAX_ITERATIONS = 2  # 最大迭代次数，避免无限循环


def search_web(query):
    """执行网络搜索，返回前5条结果"""
    url = (f"https://serpapi.com/search.json?engine=baidu&q={query}&api_key="
           f"{search_api_key}&hl=zh-CN&gl=cn")
    try:
        response = requests.get(url)
        results = response.json().get("organic_results", [])
        return "\n".join([res.get("snippet", "") for res in results[:5]])
    except Exception as e:
        return f"搜索出错: {str(e)}"


def generate_search_query(prompt):
    """生成搜索关键词"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"为回答以下问题，直接生成中文搜索关键词搜索房价相关政策，要求简洁且覆盖核心信息，包括年份，地点(精确到区)等，不包含额外的回答语句。\n问题：{prompt}"}]
    )
    return response.choices[0].message.content


def self_evaluate(prediction):
    """自我评估预测结果的潜在漏洞"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"分析以下房价预测的潜在漏洞，重点检查是否遗漏政策时效性、区域差异、数据准确性等关键因素，用简洁语言列出问题：\n{prediction}"}]
    )
    return response.choices[0].message.content


def llm_with_iteration(prompt):
    """带迭代修正的LLM处理流程"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    iterations = 0
    search_results = ""
    current_prediction = ""

    while iterations < MAX_ITERATIONS:
        # 构建增强提示
        if iterations == 0:
            # 初始轮：基于原始问题搜索
            search_query = generate_search_query(prompt)
            search_results = search_web(search_query)
            if DEBUG:
                print(f"* 第{iterations+1}轮搜索关键词: {search_query}")
                print(f"* 第{iterations+1}轮搜索结果: {search_results}")
            enhanced_prompt = f"结合以下信息回答问题：{search_results}\n问题：{prompt}\n请给出房价走势预测（接下来会上升/下降）"
            if iterations == MAX_ITERATIONS - 1:
                enhanced_prompt += "。生成最终预测结果，无需输出思维链"
            else:
                enhanced_prompt += "并输出思维链"
            enhanced_prompt += "。不要有多余的回答语句"
        else:
            # 迭代轮：基于评估结果补充搜索
            assess_result = self_evaluate(current_prediction)
            if DEBUG:
                print(f"\n* 自我评估结果:\n{assess_result}")

            # 判断是否需要继续搜索（含不确定性关键词则触发）
            if "未明确" in assess_result or "遗漏" in assess_result or "不明" in assess_result or "不确定" in assess_result or "忽略" in assess_result or "缺" in assess_result or "过度" in assess_result:
               new_search_query = generate_search_query(f"{prompt}。需补充信息：{assess_result}")
               new_search = search_web(new_search_query)
               if DEBUG:
                   print(f"* 第{iterations+1}轮补充搜索关键词: {new_search_query}")
                   print(f"* 第{iterations+1}轮补充搜索结果: {new_search}")
               search_results += "\n" + new_search
               enhanced_prompt = f"基于初始预测：{current_prediction}\n补充信息：{new_search}\n问题：{prompt}\n请修正房价走势预测，确保覆盖所有关键因素。直接回答问题，无需多余语句"

        # 生成预测
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        current_prediction = response.choices[0].message.content
        if DEBUG and iterations < MAX_ITERATIONS:
            print(f"\n# 第{iterations+1}轮预测结果: {current_prediction}")

        iterations += 1

    return current_prediction


# 使用示例
if __name__ == "__main__":
    query = "2025年Q3上海浦东严桥路房价走势如何？"
    original_stdout = sys.stdout
    with open('answer.md', 'w', encoding='utf-8') as f:
        sys.stdout = f
        final_result = llm_with_iteration(query)
        print("* 最终预测结果:\n    * " + final_result)
    sys.stdout = original_stdout
    print("\n* 结果已写入answer.md")