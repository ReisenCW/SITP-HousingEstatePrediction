import requests
import os
import sys
from datetime import datetime
import pandas_datareader.data as web
from openai import OpenAI

DEBUG = False
MAX_ITERATIONS = 2

# 设置 API 密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
search_api_key = "f97a60574d1ce7566fcf06b15c8b07bf9eeebf21d94e9b2480bc96acccadf95b"

# 映射常用指标到 pandas_datareader/FRED 支持的代码（这里是示例，真实使用时你可以替换为中国宏观数据接口）
indicator_mapping = {
    "GDP": "CHNGDPRQDSMEI",
    "CPI": "CHNCPIALLMINMEI",
    "失业率": "CHNURTOTQDSMEI",
    "M2": "CHNM2",
    "贷款利率": "CHNIRLTLT01STM",
    "LPR": "CHNLPR",  # 示例代码，实际可能需要接其他数据源
}


def search_web(query):
    url = (f"https://serpapi.com/search.json?engine=baidu&q={query}&api_key="
           f"{search_api_key}&hl=zh-CN&gl=cn")
    try:
        response = requests.get(url)
        results = response.json().get("organic_results", [])
        return "\n".join([res.get("snippet", "") for res in results[:5]])
    except Exception as e:
        return f"搜索出错: {str(e)}"


def get_openai_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def generate_search_query(prompt):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"为回答以下问题，直接生成中文搜索关键词搜索房价相关政策，要求简洁且覆盖核心信息，包括年份，地点(精确到区)等，不包含额外的回答语句。\n问题：{prompt}"}]
    )
    return response.choices[0].message.content.strip()


def self_evaluate(prediction):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"分析以下房价预测的潜在漏洞，重点检查是否遗漏政策时效性、区域差异、数据准确性等关键因素，用简洁语言列出问题：\n{prediction}"}]
    )
    return response.choices[0].message.content.strip()


def ask_needed_indicators(prompt):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"为了预测{prompt}，需要哪些宏观金融指标？请只列出简洁的中文指标名，用顿号分隔，例如：GDP、CPI、失业率、LPR"}]
    )
    return response.choices[0].message.content.strip()


def get_financial_data(indicators):
    start_date = f"{datetime.now().year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    financial_data = ""

    for ind in indicators.split("、"):
        symbol = indicator_mapping.get(ind.strip())
        if not symbol:
            financial_data += f"{ind}：暂不支持的数据\n"
            continue
        try:
            df = web.DataReader(symbol, "fred", start_date, end_date)
            latest = df.iloc[-1].values[0]
            financial_data += f"{ind}：{latest}\n"
        except Exception as e:
            financial_data += f"{ind}：获取失败（{str(e)}）\n"
    return financial_data


def llm_with_iteration(prompt):
    client = get_openai_client()
    iterations = 0
    search_results = ""
    current_prediction = ""

    while iterations < MAX_ITERATIONS:
        if iterations == 0:
            search_query = generate_search_query(prompt)
            search_results = search_web(search_query)

            indicators = ask_needed_indicators(prompt)
            financial_data = get_financial_data(indicators)

            if DEBUG:
                print(f"\n[搜索关键词]：{search_query}")
                print(f"\n[金融指标]：{indicators}")
                print(f"\n[金融数据]：\n{financial_data}")

            enhanced_prompt = (
                f"根据以下政策搜索结果与金融数据预测房价走势：\n"
                f"搜索结果：\n{search_results}\n"
                f"金融数据（截至目前）：\n{financial_data}\n"
                f"问题：{prompt}\n请直接预测接下来房价会上升或下降，并说明原因，不要输出多余语句。"
            )
        else:
            assess_result = self_evaluate(current_prediction)
            if DEBUG:
                print(f"\n[自我评估]：{assess_result}")

            if any(k in assess_result for k in ["未明确", "遗漏", "不明", "不确定", "忽略", "缺", "过度"]):
                new_query = generate_search_query(f"{prompt}。需补充信息：{assess_result}")
                new_search = search_web(new_query)
                search_results += "\n" + new_search

                enhanced_prompt = (
                    f"基于初始预测：{current_prediction}\n"
                    f"补充搜索信息：\n{new_search}\n"
                    f"问题：{prompt}\n请重新给出更准确的预测。"
                )
            else:
                break

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        current_prediction = response.choices[0].message.content.strip()

        if DEBUG:
            print(f"\n[第{iterations + 1}轮预测]：{current_prediction}")

        iterations += 1

    return current_prediction


# 示例入口
if __name__ == "__main__":
    query = "2025年Q3上海浦东严桥路房价走势如何？"
    original_stdout = sys.stdout
    with open('answer.md', 'w', encoding='utf-8') as f:
        sys.stdout = f
        result = llm_with_iteration(query)
        print("* 最终预测结果:\n    * " + result)
    sys.stdout = original_stdout
    print("\n✅ 预测结果已写入文件 answer.md")
