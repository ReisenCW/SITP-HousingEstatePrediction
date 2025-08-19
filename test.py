import os
import sys
from datetime import datetime
import pandas_datareader.data as web
from openai import OpenAI
from urllib.parse import quote
from playwright.sync_api import sync_playwright
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEBUG = True
MAX_ITERATIONS = 2

# 设置 API 密钥
api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化嵌入模型（用于将文本转为向量）
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级模型，适合本地使用
# 初始化Chroma向量数据库（存储在本地目录）
db_client = chromadb.Client(
    Settings(
        persist_directory="./local_search_db",  # 本地存储路径
        anonymized_telemetry=False  # 关闭匿名统计
    )
)
# 创建/获取集合（类似数据库表）
collection = db_client.get_or_create_collection(
    name="housing_search_results",
    metadata={"description": "存储房价相关搜索的查询和结果"}
)

# 映射常用指标到 pandas_datareader/FRED 支持的代码
indicator_mapping = {
    "GDP": "CHNGDPRQDSMEI",
    "CPI": "CHNCPIALLMINMEI",
    "失业率": "CHNURTOTQDSMEI",
    "M2": "CHNM2",
    "贷款利率": "CHNIRLTLT01STM",
    "LPR": "CHNLPR",
}


def local_search(query, threshold=0.7):
    """
    从本地向量库检索相似内容
    :param query: 新查询词
    :param threshold: 相似度阈值（超过此值认为相关）
    :return: 相关结果文本列表及元数据（空列表表示无匹配）
    """
    # 生成查询向量
    query_embedding = embedding_model.encode([query])[0].tolist()

    # 检索相似结果（返回前3条最相似的）
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # 过滤出相似度高于阈值的结果（距离越小越相似，此处转换为相似度）
    matched_results = []
    for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
    ):
        similarity = 1 - distance  # Chroma返回的是距离，转为相似度
        if similarity > threshold:
            matched_results.append({
                "document": doc,
                "metadata": metadata,
                "similarity": similarity
            })

    return matched_results


def store_to_local(query, search_result, duplicate_threshold=0.9):
    """
    存储新的搜索结果到本地向量库，避免重复
    :param query: 搜索查询词
    :param search_result: 网络搜索返回的结果文本
    :param duplicate_threshold: 去重阈值（超过此值认为重复）
    """
    # 生成结果文本的向量
    result_embedding = embedding_model.encode([search_result])[0].tolist()

    # 检查是否与已有结果重复（检索最相似的已有结果）
    existing = collection.query(
        query_embeddings=[result_embedding],
        n_results=1,
        include=["distances"]
    )

    # 若最相似结果的相似度低于阈值，则存储新结果
    if not existing["distances"][0] or (
            1 - existing["distances"][0][0]) < duplicate_threshold:
        collection.add(
            documents=[search_result],  # 存储搜索结果文本
            metadatas=[
                {"query": query, "timestamp": datetime.now().isoformat()}],
            # 附加元数据（查询词、时间）
            embeddings=[result_embedding],  # 存储向量
            ids=[f"id_{datetime.now().timestamp()}"]  # 唯一ID（用时间戳避免重复）
        )
        if DEBUG:
            print("已存储新结果到本地向量库")
    else:
        if DEBUG:
            print("结果与本地已有内容重复，未存储")


def retrieve_info(url, browser):
    """在新标签页打开链接并获取信息"""
    try:
        # 打开新标签页
        page = browser.new_page(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
        )

        # 导航到目标页面，设置超时时间
        page.goto(url, timeout=20000)
        page.wait_for_selector('div.txtinfos', timeout=5000)
        paragraphs = page.locator('div.txtinfos p')
        text = []
        for i in range(paragraphs.count()):
            text.append(paragraphs.nth(i).text_content().strip())

        # 获取页面标题
        title = page.title()

        # 关闭当前标签页
        page.close()
        return {
            "title": title,
            "url": url,
            "content": ''.join(text)
        }
    except Exception as e:
        return {
            "title": "获取失败",
            "url": url,
            "content": f"获取详情失败: {str(e)}"
        }


def filter_similar_results(results, threshold=0.8):
    """
    过滤相似结果（基于标题向量相似度）
    :param results: 结构化结果列表（含title、url、content）
    :param threshold: 相似度阈值（超过此值视为相似）
    :return: 去重后的结果列表
    """
    if len(results) <= 1:
        return results  # 只有1条结果时无需过滤

    # 对标题进行向量编码（标题更能反映核心内容，效率更高）
    titles = [r["title"] for r in results]
    title_embeddings = embedding_model.encode(titles)  # 形状：[n, 384]

    # 计算相似度矩阵（n x n）
    similarity_matrix = cosine_similarity(title_embeddings)

    # 标记需要保留的结果（默认保留第一条，过滤后续相似结果）
    keep = [True] * len(results)
    for i in range(len(results)):
        if not keep[i]:
            continue  # 已被标记为丢弃，跳过
        # 比较i与后续所有结果的相似度
        for j in range(i + 1, len(results)):
            if similarity_matrix[i][j] > threshold:
                keep[j] = False  # 标记为丢弃
                if DEBUG:
                    print(
                        f"过滤相似结果：{results[j]['title']}（与{results[i]['title']}相似度：{similarity_matrix[i][j]:.2f}）")

    # 保留未被标记的结果
    filtered = [results[i] for i in range(len(results)) if keep[i]]
    return filtered

def search_web(query, browser, search_page):
    """使用现有浏览器和搜索页面进行搜索"""
    # 构造东方财富网搜索URL
    encoded_query = quote(query)
    url = f"https://so.eastmoney.com/news/s?keyword={encoded_query}"
    try:
        # 直接修改URL进行搜索
        search_page.goto(url, timeout=10000)
        # 等待搜索结果加载
        search_page.wait_for_selector('div.news_list', timeout=5000)
        # 获取新闻列表
        results = []
        news_items = search_page.locator('div.news_list div.news_item')

        # 遍历前5条结果
        for i in range(min(news_items.count(), 10)):
            # 获取新闻链接和标题
            a_element = news_items.nth(i).locator('div.news_item_url a')
            news_url = a_element.get_attribute('href')
            news_title = a_element.text_content().strip() if a_element.text_content() else "无标题"

            if news_url:
                # 在新标签页打开并获取信息
                detail = retrieve_info(news_url, browser)
                results.append(detail)
        filtered_results = filter_similar_results(results, threshold=0.8)

        if DEBUG:
            print("\n===== 过滤后网络搜索结果 =====")
            for i, res in enumerate(filtered_results, 1):
                print(f"结果 {i}：{res['title']}（{res['url']}）")

        return filtered_results

    except Exception as e:
        return [{"title": "搜索出错", "url": url,
                 "content": f"东方财富网搜索出错: {str(e)}"}]


def search_with_local_priority(query, browser, search_page):
    """优先从本地检索，同时执行网络搜索以获取新内容，合并结果"""
    # 1. 先查本地
    local_results = local_search(query)

    # DEBUG模式下打印本地结果信息
    if DEBUG and local_results:
        print("\n===== 本地向量库搜索结果 =====")
        for i, result in enumerate(local_results, 1):
            query_text = result["metadata"].get("query", "未知查询")
            timestamp = result["metadata"].get("timestamp", "未知时间")
            print(f"本地结果 {i}:")
            print(f"  关联查询: {query_text}")
            print(f"  存储时间: {timestamp}")
            print(f"  相似度: {result['similarity']:.2f}")
            print(f"  内容摘要: {result['document'][:100]}...\n")

    # 2. 执行网络搜索（无论本地是否有结果）
    web_results = search_web(query, browser, search_page)

    # DEBUG模式下打印网络结果信息
    if DEBUG and web_results:
        print("\n===== 网络搜索结果 =====")
        for i, result in enumerate(web_results, 1):
            print(f"网络结果 {i}:")
            print(f"  标题: {result['title']}")
            print(f"  URL: {result['url']}\n")

    # 3. 提取并合并所有结果文本
    local_texts = [r["document"] for r in local_results]
    web_texts = [r["content"] for r in web_results]
    all_texts = local_texts + web_texts
    combined_result = "\n".join(all_texts) if all_texts else "未找到相关结果"

    # 4. 存储新的网络结果到本地（自动去重）
    for web_result in web_results:
        if web_result["content"] and "未找到相关结果" not in web_result[
            "content"]:
            store_to_local(query, web_result["content"])

    return combined_result


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
            f"请根据下列房价预测问题，提取最简明、最核心的中文搜索关键词，要求：\n - 仅包含年份、城市、区县、街道/小区等地名信息； \n - 不包含“走势”、“预测”、“政策”等限定性或无关词汇；\n - 不输出任何解释、标点或额外语句，仅输出关键词，关键词之间用空格分隔。\n 示例：问 2025年Q1上海黄浦区小南门房价走势如何？→ 输出 2025 上海 黄浦 小南门。\n 问题：{prompt}"}]
    )
    return response.choices[0].message.content.strip()


def self_evaluate(prediction):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content":
            f"请针对以下房价预测，列出所有潜在漏洞和不足，要求：\n - 只输出问题清单，不要复述预测内容；\n - 按“政策时效性”、“区域代表性”、“数据支撑”、“逻辑链条”、“风险遗漏”等分类，每类下可有多条；\n - 语言简洁明了。\n\n预测内容：\n{prediction}"}]
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

    # 启动浏览器并保持实例
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--enable-features=NetworkService,NetworkServiceInProcess"],
        )
        # 创建搜索专用页面
        search_page = browser.new_page(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
        )

        try:
            while iterations < MAX_ITERATIONS:
                if iterations == 0:
                    search_query = generate_search_query(prompt)
                    search_results = search_with_local_priority(search_query,
                                                                browser,
                                                                search_page)

                    enhanced_prompt = (
                        f"当前时间：{datetime.now().strftime('%Y-%m-%d')}\n"
                        f"请基于下列政策和市场信息，预测“{prompt}”的房价走势。要求：\n"
                        f"1. 先用一句话给出明确预测结论（如“预计2025年Q3上海浦东严桥路房价将温和上涨，涨幅约xx-xx”, 置信度为xx）。\n"
                        f"2. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果1...”）。\n"
                        f"3. 简要列出可能的风险或不确定性因素。\n"
                        f"4. 输出内容分为“预测结论”、“主要依据”、“风险提示”三部分，禁止输出与预测无关的内容。\n\n"
                        f"搜索结果：\n{search_results}\n"
                    )
                else:
                    assess_result = self_evaluate(current_prediction)
                    if DEBUG:
                        print(f"\n[自我评估]：{assess_result}")

                    if any(k in assess_result for k in
                           ["未明确", "遗漏", "不明", "不确定", "忽略", "缺",
                            "过度"]):
                        new_query = generate_search_query(
                            f"请根据对“{prompt}”的预测自我评估结果：{assess_result}，明确指出还需补充哪些关键信息或数据（如政策时效、区域供需、最新市场数据、具体楼盘信息等），并将这些补充点转化为最简明、最核心的中文检索关键词。要求：仅输出关键词，关键词之间用空格分隔，不输出任何解释或多余语句。")
                        new_search = search_with_local_priority(new_query,
                                                                browser,
                                                                search_page)
                        search_results += "\n" + new_search

                        enhanced_prompt = (
                            f"基于初始预测：{current_prediction}\n"
                            f"补充搜索信息：\n{new_search}\n"
                            f"问题：{prompt}\n请重新给出更准确的预测。"
                        )
                        if DEBUG:
                            print(f"检索式: {new_query} ")
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
        finally:
            # 迭代结束后关闭所有页面和浏览器
            search_page.close()
            browser.close()

    return current_prediction


if __name__ == "__main__":
    query = "2025年Q3上海浦东严桥路房价走势如何？"
    original_stdout = sys.stdout
    with open('answer.md', 'w', encoding='utf-8') as f:
        sys.stdout = f
        result = llm_with_iteration(query)
        print("* 最终预测结果:\n    * " + result)
    sys.stdout = original_stdout
    print("\n✅ 预测结果已写入文件 answer.md")