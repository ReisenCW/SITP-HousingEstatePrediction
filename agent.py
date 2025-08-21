import os
import sys
from datetime import datetime
import pandas_datareader.data as web
from openai import OpenAI
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma as LangchainChroma
from config import Config

# 初始化langchain的Chroma向量数据库
if not os.path.exists(Config.PERSIST_DIRECTORY):
    os.makedirs(Config.PERSIST_DIRECTORY)

# 兼容langchain的embedding接口
embedding_func = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
vectorstore = LangchainChroma(
    embedding_function=embedding_func,
    persist_directory=Config.PERSIST_DIRECTORY,
    collection_name="housing_search_results"
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


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.embedding_func = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.vectorstore = LangchainChroma(
            embedding_function=self.embedding_func,
            persist_directory=config.PERSIST_DIRECTORY,
            collection_name="housing_search_results"
        )

    def store_to_local(self, query, search_result):
        self.vectorstore.add_texts(
            [search_result],
            metadatas=[{"query": query, "timestamp": datetime.now().isoformat()}]
        )
        if self.config.DEBUG:
            print("已存储新结果到本地向量库")

    def local_search(self, query, threshold=0.7):
        results = self.vectorstore.similarity_search_with_score(query, k=self.config.LOCAL_SEARCH_TOP_K)
        matched_results = []
        for doc, score in results:
            similarity = 1 - score if score is not None else 0
            if self.config.DEBUG:
                print(f"相似度: {similarity}")
            if similarity > threshold:
                matched_results.append({
                    "document": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "similarity": similarity
                })
                if self.config.DEBUG:
                    print(f"本地搜索匹配结果: {matched_results} \n")
        return matched_results

    def retrieve_info(self, url, browser):
        try:
            page = browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
            )
            page.goto(url, timeout=10000)
            title = page.title()
            content = page.locator('div.txtinfos').text_content()
            content = content.strip() if content else ""
            page.close()
            return {
                "title": title if title else "无标题",
                "url": url,
                "content": content if content else "未能提取正文内容"
            }
        except Exception as e:
            return {
                "title": "获取失败",
                "url": url,
                "content": f"获取详情失败: {str(e)}"
            }

    def filter_similar_results(self, results, threshold=0.8):
        if len(results) <= 1:
            return results
        titles = [r["title"] for r in results]
        title_embeddings = self.embedding_func.embed_documents(titles)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(title_embeddings)
        keep = [True] * len(results)
        for i in range(len(results)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(results)):
                if similarity_matrix[i][j] > threshold:
                    keep[j] = False
                    if self.config.DEBUG:
                        print(f"过滤相似结果：{results[j]['title']}（与{results[i]['title']}相似度：{similarity_matrix[i][j]:.2f}）")
        filtered = [results[i] for i in range(len(results)) if keep[i]]
        return filtered

    def search_web(self, query, browser, search_page):
        from urllib.parse import quote
        encoded_query = quote(query)
        url = f"https://so.eastmoney.com/news/s?keyword={encoded_query}"
        try:
            search_page.goto(url, timeout=10000)
            search_page.wait_for_selector('div.news_list', timeout=5000)
            results = []
            news_items = search_page.locator('div.news_list div.news_item')
            news_count = news_items.count()
            for i in range(min(news_count, self.config.WEB_SEARCH_TOP_K)):
                a_element = news_items.nth(i).locator('div.news_item_url a')
                news_url = a_element.get_attribute('href')
                news_title = a_element.text_content().strip() if a_element.text_content() else "无标题"
                if news_url:
                    detail = self.retrieve_info(news_url, browser)
                    results.append(detail)
            filtered_results = self.filter_similar_results(results, threshold=0.8)
            for res in filtered_results:
                doc_text = f"{res.get('title', '')}\n{res.get('content', '')}"
                self.store_to_local(query, doc_text)
            if self.config.DEBUG:
                print("\n===== 过滤后网络搜索结果 =====")
                for i, res in enumerate(filtered_results, 1):
                    print(f"结果 {i}：{res.get('title', '')}（{res.get('url', '')}）")
            return filtered_results
        except Exception as e:
            return [{"title": "搜索出错", "url": url, "content": f"东方财富网搜索出错: {str(e)}"}]

    def extract_relevant_chunks(self, query, texts, top_k=8, chunk_size=200):
        """
        将每条文本分割为片段，计算与query的相似度，返回最相关的top_k片段。
        """
        import re
        # 1. 分割为片段（按段落或句子，长度不超过chunk_size）
        chunks = []
        for idx, text in enumerate(texts):
            # 按段落分割
            for para in re.split(r'\n+', text):
                para = para.strip()
                if not para:
                    continue
                # 长段落再按句子分割
                if len(para) > chunk_size:
                    sentences = re.split(r'[。！？!?.]', para)
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            chunks.append((f"doc{idx+1}", sent))
                else:
                    chunks.append((f"doc{idx+1}", para))
        if not chunks:
            return []
        # 2. 计算embedding
        chunk_texts = [c[1] for c in chunks]
        chunk_embs = self.embedding_func.embed_documents(chunk_texts)
        query_emb = self.embedding_func.embed_query(query)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([query_emb], chunk_embs)[0]
        # 3. 取top_k相关片段
        top_indices = sims.argsort()[-top_k:][::-1]
        relevant_chunks = [f"[{chunks[i][0]}] {chunks[i][1]}" for i in top_indices]
        return relevant_chunks

    def search_with_local_priority(self, query, browser, search_page):
        local_results = self.local_search(query)
        if self.config.DEBUG and local_results:
            print("\n===== 本地向量库搜索结果 =====")
            for i, result in enumerate(local_results, 1):
                query_text = result["metadata"].get("query", "未知查询")
                timestamp = result["metadata"].get("timestamp", "未知时间")
                print(f"本地结果 {i}:")
                print(f"  关联查询: {query_text}")
                print(f"  存储时间: {timestamp}")
                print(f"  相似度: {result['similarity']:.2f}")
                print(f"  内容摘要: {result['document'][:100]}...\n")
        web_results = self.search_web(query, browser, search_page)
        if self.config.DEBUG and web_results:
            print("\n===== 网络搜索结果 =====")
            for i, result in enumerate(web_results, 1):
                print(f"网络结果 {i}:")
                print(f"  标题: {result['title']}")
                print(f"  URL: {result['url']}\n")
        local_texts = [r["document"] for r in local_results]
        web_texts = [r["content"] for r in web_results]
        all_texts = local_texts + web_texts
        # RAG片段级筛选
        relevant_chunks = self.extract_relevant_chunks(query, all_texts, top_k=8, chunk_size=200)
        combined_result = "\n".join(relevant_chunks) if relevant_chunks else "未找到相关结果"
        return combined_result

    def generate_search_query(self, prompt):
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content":
                f"请根据下列房价预测问题，提取最简明、最核心的中文搜索关键词，要求：\n - 仅包含年份、城市、区县、街道/小区等地名信息； \n - 不包含“走势”、“预测”、“政策”等限定性或无关词汇；\n - 不输出任何解释、标点或额外语句，仅输出关键词，关键词之间用空格分隔。\n 示例：问 2025年Q1上海黄浦区小南门房价走势如何？→ 输出 2025 上海 黄浦 小南门。\n 问题：{prompt}"}]
        )
        return response.choices[0].message.content.strip()

    def self_evaluate(self, prediction):
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content":
                f"请针对以下房价预测，列出所有潜在漏洞和不足，要求：\n - 只输出问题清单，不要复述预测内容；\n - 按“政策时效性”、“区域代表性”、“数据支撑”、“逻辑链条”、“风险遗漏”等分类，每类下可有多条；\n - 语言简洁明了。\n\n预测内容：\n{prediction}"}]
        )
        return response.choices[0].message.content.strip()

    def ask_needed_indicators(self, prompt):
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content":
                f"为了预测{prompt}，需要哪些宏观金融指标？请只列出简洁的中文指标名，用顿号分隔，例如：GDP、CPI、失业率、LPR"}]
        )
        return response.choices[0].message.content.strip()

    def get_financial_data(self, indicators):
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

    def llm_with_iteration(self, prompt):
        iterations = 0
        search_results = ""
        current_prediction = ""
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--enable-features=NetworkService,NetworkServiceInProcess"],
            )
            search_page = browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
            )
            try:
                while iterations < self.config.MAX_ITERATIONS:
                    if iterations == 0:
                        search_query = self.generate_search_query(prompt)
                        search_results = self.search_with_local_priority(search_query,
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
                        assess_result = self.self_evaluate(current_prediction)
                        if self.config.DEBUG:
                            print(f"\n[自我评估]：{assess_result}")
                        if any(k in assess_result for k in
                               ["未明确", "遗漏", "不明", "不确定", "忽略", "缺", "过度"]):
                            new_query = self.generate_search_query(
                                f"请根据对“{prompt}”的预测自我评估结果：{assess_result}，明确指出还需补充哪些关键信息或数据（如政策时效、区域供需、最新市场数据、具体楼盘信息等），并将这些补充点转化为最简明、最核心的中文检索关键词。要求：仅输出关键词，关键词之间用空格分隔，不输出任何解释或多余语句。")
                            new_search = self.search_with_local_priority(new_query,
                                                                         browser,
                                                                         search_page)
                            search_results += "\n" + new_search
                            enhanced_prompt = (
                                f"基于初始预测：{current_prediction}\n"
                                f"补充搜索信息：\n{new_search}\n"
                                f"问题：{prompt}\n请重新给出更准确的预测。"
                            )
                            if self.config.DEBUG:
                                print(f"检索式: {new_query} ")
                        else:
                            break
                    response = self.client.chat.completions.create(
                        model="qwen-plus",
                        messages=[{"role": "user", "content": enhanced_prompt}]
                    )
                    current_prediction = response.choices[0].message.content.strip()
                    if self.config.DEBUG:
                        print(f"\n[第{iterations + 1}轮预测]：{current_prediction}")
                    iterations += 1
            finally:
                search_page.close()
                browser.close()
        return current_prediction


if __name__ == "__main__":
    query = "2025年Q3上海浦东严桥路房价走势如何？"
    agent = Agent(Config())
    original_stdout = sys.stdout
    with open('answer.md', 'w', encoding='utf-8') as f:
        sys.stdout = f
        result = agent.llm_with_iteration(query)
        print("* 最终预测结果:\n    * " + result)
    sys.stdout = original_stdout
    print("\n✅ 预测结果已写入文件 answer.md")

    # 打印本地向量数据库所有内容
    # all_docs = vectorstore._collection.get()
    # for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    #     print(doc)
    #     print(meta)

    # 本周进度：
    # 1.网上搜索信息保存：由于网页是动态加载的,html中body无网页内容,因此不再使用require和bs4,改用playwright自动化
    # 2.网上搜索信息保存到本地Chroma向量数据库(Langchain框架), 下次搜索时先在本地搜索,再在网络搜索
    # 3.对于检索到的消息,进行分片,通过向量化取出与问题有关的内容作为上下文
    # 4.优化了prompt
    # 5.将函数和常量进行封装(Config和Agent类),使代码结构更清晰
