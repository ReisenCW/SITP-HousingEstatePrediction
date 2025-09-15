import os
import sys
import re
import asyncio
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from openai import OpenAI
from playwright.async_api import async_playwright
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma as LangchainChroma
from config import Config
from expLib import ExpLib
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
# indicator_mapping = {
#     "GDP": "CHNGDPRQDSMEI",
#     "CPI": "CHNCPIALLMINMEI",
#     "失业率": "CHNURTOTQDSMEI",
#     "M2": "CHNM2",
#     "贷款利率": "CHNIRLTLT01STM",
#     "LPR": "CHNLPR",
# }


class Agent:
    def debug_log(self, msg):
        with open('log.md', 'a', encoding='utf-8') as logf:
            logf.write(str(msg) + '\n')
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
        self.exp_lib = ExpLib(embedding_func=self.embedding_func)

    async def search_site(self, query, browser, search_page=None, site="eastmoney", top_k=5, cutoff_time=None):
        """
        site: "eastmoney" 或 "shfgj"
        browser: playwright异步browser对象
        search_page: 东方财富网需传入已打开的search_page，房管局无需
        top_k: 返回条数
        cutoff_time: 时间过滤
        """
        if site == "eastmoney":
            encoded_query = quote(query)
            url = f"https://so.eastmoney.com/news/s?keyword={encoded_query}"
            results = []
            seen_urls = set()
            page_num = 1
            max_page_num = 1
            try:
                await search_page.goto(url, timeout=5000)
                await search_page.wait_for_selector('div.news_list', timeout=3000)
                # 获取div.c_pager下所有a标签
                pager_a_list = await search_page.locator('div.c_pager a').all()
                page_nums = []
                for a in pager_a_list:
                    try:
                        text = await a.text_content()
                        if text and text.isdigit():
                            page_nums.append(int(text))
                    except Exception:
                        continue
                if page_nums:
                    max_page_num = max(page_nums)
                else:
                    max_page_num = 1
            except Exception as e:
                if self.config.DEBUG:
                    self.debug_log(f"获取最大页码失败: {e}")
                max_page_num = 1
            while len(results) < top_k and page_num <= max_page_num:
                if page_num > 1:
                    try:
                        textbox = search_page.locator('input[type="text"]')
                        await textbox.nth(1).click()
                        await textbox.nth(1).fill(str(page_num))
                        await textbox.nth(1).press('Enter')
                        await search_page.wait_for_selector('div.news_list', timeout=3000)
                    except Exception as e:
                        if self.config.DEBUG:
                            self.debug_log(f"翻页失败: {e}")
                        break
                await search_page.wait_for_selector('div.news_list', timeout=3000)
                news_items = search_page.locator('div.news_list div.news_item')
                news_count = await news_items.count()
                news_meta_list = []
                for i in range(news_count):
                    # 标题
                    title_a = news_items.nth(i).locator('div.news_item_t a')
                    news_title = await title_a.text_content() or "无标题"
                    news_title = news_title.strip()
                    # url
                    a_element = news_items.nth(i).locator('div.news_item_url a')
                    news_url = await a_element.get_attribute('href')
                    timestamp = None
                    try:
                        timestamp = await news_items.nth(i).locator('div.news_item_c').text_content()
                        if timestamp and len(timestamp) >= 10:
                            timestamp = timestamp[:10]
                    except Exception as e:
                        self.debug_log(f"获取时间戳失败: {news_title}, 错误: {e}")
                    if not news_url or news_url in seen_urls:
                        continue
                    if cutoff_time and timestamp and timestamp > cutoff_time:
                        if self.config.DEBUG:
                            self.debug_log(f"跳过超出时间的新闻: {news_title}, {timestamp}")
                        continue
                    seen_urls.add(news_url)
                    news_meta_list.append({
                        "url": news_url,
                        "title": news_title,
                        "timestamp": timestamp
                    })
                async def fetch_detail(meta):
                    detail = await self.retrieve_info(meta["url"], browser) if meta["url"] else None
                    if detail:
                        # 只保留正文，标题和时间只用meta里的
                        return {
                            "title": meta["title"],
                            "timestamp": meta["timestamp"],
                            "url": meta["url"],
                            "content": detail.get("content", "")
                        }
                    return None
                details = await asyncio.gather(*(fetch_detail(meta) for meta in news_meta_list))
                for detail in details:
                    if detail:
                        results.append(detail)
                        if len(results) >= top_k:
                            break
                page_num += 1
            return results
        elif site == "shfgj":
            return await self.search_shfgj(query, browser, max_results=top_k, cutoff_time=cutoff_time)
        elif site == "souhu":
            return await self.search_souhu(query, browser, max_results=top_k, cutoff_time=cutoff_time)
        else:
            raise ValueError(f"未知site: {site}")

    def store_to_local(self, query, search_result, timestamp=None, title=None, threshold=0.05):
        # 检查是否为重复内容（与库中已有内容距离分数很小则跳过）
        existing = self.vectorstore.similarity_search_with_score(search_result, k=3)
        for doc, score in existing: # score为距离分数，越小说明越相似
            if score is not None and score < threshold:
                if self.config.DEBUG:
                    self.debug_log(f"查重：与已有内容距离分数{score:.4f}，跳过存储。")
                return False
        meta = {"query": query}
        if timestamp:
            meta["timestamp"] = timestamp
        else:
            meta["timestamp"] = datetime.now().isoformat()
        if title:
            meta["title"] = title
        self.vectorstore.add_texts(
            [search_result],
            metadatas=[meta]
        )
        if self.config.DEBUG:
            self.debug_log("已存储新结果到本地向量库")
        return True

    def local_search(self, query, threshold=0.3, cutoff_time=None):
        results = self.vectorstore.similarity_search_with_score(query, k=self.config.LOCAL_SEARCH_TOP_K)
        matched_results = []
        if self.config.DEBUG and cutoff_time:
            self.debug_log(f"[本地检索] 时间过滤cutoff_time: {cutoff_time}")
        for doc, score in results:
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            doc_time = meta.get('timestamp', '')[:10] if meta.get('timestamp') else ''
            if cutoff_time and doc_time and doc_time > cutoff_time:
                if self.config.DEBUG:
                    self.debug_log(f"过滤掉超出时间的内容: 标题={meta.get('title','')}, url={meta.get('url','')}, time={doc_time}")
                continue
            if self.config.DEBUG:
                self.debug_log(f"标题: {meta.get('title','')} , 距离分数: {score}")
            if score is not None and score < threshold:
                matched_results.append({
                    "document": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": meta,
                    "distance": score
                })
        if self.config.DEBUG:
            self.debug_log("[本地检索] 结束 \n\n")
        return matched_results

    async def retrieve_info(self, url, browser):
        try:
            page = await browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
            )
            await page.goto(url, timeout=5000)
            title = await page.title()
            content = await page.locator('div.txtinfos').text_content()
            content = content.strip() if content else ""
            await page.close()
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
        urls = [r.get("url","") for r in results]
        times = [r.get("timestamp","") for r in results]
        title_embeddings = self.embedding_func.embed_documents(titles)
        similarity_matrix = cosine_similarity(title_embeddings)
        keep = [True] * len(results)
        for i in range(len(results)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(results)):
                if similarity_matrix[i][j] > threshold:
                    keep[j] = False
                    if self.config.DEBUG:
                        self.debug_log(f"过滤相似结果：{results[j]['title']}（与{results[i]['title']}相似度：{similarity_matrix[i][j]:.2f}）")
                        self.debug_log(f"  url1={urls[i]}  url2={urls[j]}")
        filtered = [results[i] for i in range(len(results)) if keep[i]]
        if self.config.DEBUG:
            self.debug_log(f"相似度过滤后剩余{len(filtered)}条")
        return filtered

    async def search_web(self, query, browser, search_page, cutoff_time=None, max_pages=10):
        encoded_query = quote(query)
        url = f"https://so.eastmoney.com/news/s?keyword={encoded_query}"
            
        eastmoney_k = self.config.EAST_SEARCH_TOP_K
        shfgj_k = self.config.SHFGJ_SEARCH_TOP_K
        souhu_k = self.config.SOUHU_SEARCH_TOP_K
        if self.config.DEBUG and cutoff_time:
            self.debug_log(f"[网络检索] 时间过滤cutoff_time: {cutoff_time}")
        eastmoney_results = await self.search_site(query, browser, search_page=search_page, site="eastmoney", top_k=eastmoney_k, cutoff_time=cutoff_time)
        shfgj_results = await self.search_site(query, browser, site="shfgj", top_k=shfgj_k, cutoff_time=cutoff_time)
        souhu_results = await self.search_site(query, browser, site="souhu", top_k=souhu_k, cutoff_time=cutoff_time)
        all_results = eastmoney_results + shfgj_results + souhu_results
        # 严格过滤时间，只保留cutoff_time之前的内容
        if cutoff_time:
            all_results = [r for r in all_results if r.get('timestamp') and r.get('timestamp') <= cutoff_time]
        filtered_results = self.filter_similar_results(all_results, threshold=0.8)
        for res in filtered_results:
            doc_text = f"{res.get('title', '')}\n{res.get('content', '')}"
            self.store_to_local(query, doc_text, timestamp=res.get('timestamp', None), title=res.get('title', ''))
        if self.config.DEBUG:
            self.debug_log("\n===== 过滤后网络搜索结果 =====")
            for i, res in enumerate(filtered_results, 1):
                self.debug_log(f"结果 {i}：{res.get('title', '')}("
                      f"{res.get('url', '')}) 时间: {res.get('timestamp','')}")
            self.debug_log("\n最终喂给LLM的文章：")
            for i, res in enumerate(filtered_results, 1):
                self.debug_log(f"[{i}] {res.get('title','')}\n")
        return filtered_results
    
    async def search_shfgj(self, query, browser, max_results=3, cutoff_time=None):
        results = []
        encoded_query = quote(query)
        url = f"https://fgj.sh.gov.cn/websearch.html#search/query={encoded_query}|input_type=Input|suggest_order=-1"
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=20000)
            page_num = 1
            while page_num <= 100 and len(results) < max_results:
                await page.wait_for_selector('div#maya-search-result', timeout=12000)
                items = await page.locator('div#maya-search-result div.maya-result-item.clearfix').all()
                for item in items:
                    if len(results) >= max_results:
                        break
                    a_tag = item.locator('a.doc-title')
                    news_url = await a_tag.get_attribute('href')
                    news_title = await a_tag.text_content() or "无标题"
                    news_title = news_title.strip()
                    date_spans = item.locator('div.doc-content span.doc-date')
                    timestamp = None
                    try:
                        # 只取第一个span.doc-date
                        if await date_spans.count() > 0:
                            timestamp_raw = await date_spans.nth(0).text_content()
                            if timestamp_raw and len(timestamp_raw) >= 10:
                                timestamp = timestamp_raw[:10]
                        else:
                            timestamp = None
                    except Exception:
                        self.debug_log(f"获取时间戳失败: {news_title}")
                    if not news_url:
                        continue
                    if cutoff_time and timestamp and timestamp > cutoff_time:
                        if self.config.DEBUG:
                            self.debug_log(f"跳过超出时间的房管局新闻: {news_title}, {timestamp}")
                        continue
                    # 抓正文
                    detail = await self.retrieve_shfgj_detail(news_url, browser)
                    if detail:
                        # 只保留正文，标题和时间只用列表页的
                        results.append({
                            "title": news_title,
                            "timestamp": timestamp,
                            "url": news_url,
                            "content": detail.get("content", "")
                        })
                # 翻页：点击“»”按钮
                try:
                    next_btn = page.locator('ul.pagination li:last-child span[title="下一页"]')
                    if await next_btn.count() > 0:
                        await next_btn.click()
                        page_num += 1
                        await asyncio.sleep(1)
                    else:
                        break
                except Exception as e:
                    self.debug_log(f"房管局翻页失败: {e}")
                    break
        except Exception as e:
            if self.config.DEBUG:
                self.debug_log(f"房管局搜索异常: {e}")
        await page.close()
        return results

    async def retrieve_shfgj_detail(self, url, browser):
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=8000)
            await page.wait_for_selector('div.Article', timeout=5000)
            title = await page.locator('div.Article h2#ivs_title').text_content()
            content_ps = await page.locator('div#ivs_content p').all()
            if not content_ps or len(content_ps) == 0:
                await page.close()
                return {"title": title or "无标题", "url": url, "content": "页面无正文内容（ivs_content下无p标签）"}
            content = await page.locator('div#ivs_content').text_content()
            content = content.strip() if content else ""
            await page.close()
            return {"title": title or "无标题", "url": url, "content": content.strip() if content else "未能提取正文内容"}
        except Exception as e:
            return {"title": "获取失败", "url": url, "content": f"房管局正文抓取失败: {str(e)}"}

    async def search_souhu(self, query, browser, max_results=3, cutoff_time=None):
        results = []
        encoded_query = quote(query)
        url = f"https://search.sohu.com/?queryType=outside&keyword={encoded_query}&spm=smpc.csrpage.0.0.1756974578111gLBpDfV"
        page = await browser.new_page()
        seen_urls = set()
        try:
            await page.goto(url, timeout=8000)
            last_height = await page.evaluate('document.body.scrollHeight')
            scroll_tries = 0
            while len(results) < max_results and scroll_tries < 20:
                # 抓取当前页面所有候选
                news_list = page.locator('div.search-content-left-cards')
                cards_plain = news_list.locator('div.cards-small-plain')
                cards_img = news_list.locator('div.cards-small-img')
                candidates = []
                # plain卡片
                count_plain = await cards_plain.count()
                for i in range(count_plain):
                    card = cards_plain.nth(i)
                    a_tag = card.locator('h4.plain-title a')
                    news_url = await a_tag.get_attribute('href')
                    news_title = await a_tag.text_content() or "无标题"
                    news_title = news_title.strip()
                    time_p = card.locator('p.plain-content-comm')
                    time_text = None
                    if await time_p.count() > 0:
                        p_html = await time_p.nth(0).inner_html()
                        import re
                        time_texts = re.split(r'<a[^>]*>.*?</a>', p_html)
                        date_match = None
                        for t in time_texts:
                            date_match = re.search(r'(20\d{2}-\d{1,2}-\d{1,2})', t)
                            if date_match:
                                time_text = date_match.group(1)
                                break
                    if not news_url or news_url in seen_urls:
                        continue
                    if cutoff_time and time_text and time_text > cutoff_time:
                        continue
                    seen_urls.add(news_url)
                    candidates.append({
                        "url": news_url,
                        "title": news_title,
                        "timestamp": time_text
                    })
                # img卡片
                count_img = await cards_img.count()
                for i in range(count_img):
                    card = cards_img.nth(i)
                    a_tag = card.locator('div.cards-content-title a')
                    news_url = await a_tag.get_attribute('href')
                    news_title = await a_tag.text_content() or "无标题"
                    news_title = news_title.strip()
                    time_p = card.locator('p.plain-content-comm')
                    time_text = None
                    if await time_p.count() > 0:
                        p_html = await time_p.nth(0).inner_html()
                        import re
                        time_texts = re.split(r'<a[^>]*>.*?</a>', p_html)
                        date_match = None
                        for t in time_texts:
                            date_match = re.search(r'(20\d{2}-\d{1,2}-\d{1,2})', t)
                            if date_match:
                                time_text = date_match.group(1)
                                break
                    if not news_url or news_url in seen_urls:
                        continue
                    if cutoff_time and time_text and time_text > cutoff_time:
                        continue
                    seen_urls.add(news_url)
                    candidates.append({
                        "url": news_url,
                        "title": news_title,
                        "timestamp": time_text
                    })
                # 对本轮所有候选逐条抓正文
                for meta in candidates:
                    if len(results) >= max_results:
                        break
                    detail = await self.retrieve_souhu_detail(meta["url"], browser)
                    if detail:
                        results.append({
                            "url": meta["url"],
                            "title": meta["title"],
                            "timestamp": meta["timestamp"],
                            "content": detail.get("content", "")
                        })
                # 滚动页面
                if len(results) < max_results:
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await asyncio.sleep(1.2)
                    new_height = await page.evaluate('document.body.scrollHeight')
                    if new_height == last_height:
                        scroll_tries += 1
                    else:
                        scroll_tries = 0
                        last_height = new_height
            await page.close()
        except Exception as e:
            if self.config.DEBUG:
                self.debug_log(f"搜狐搜索异常: {e}")
            await page.close()
        return results

    async def retrieve_souhu_detail(self, url, browser):
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=8000)
            # 找到div.text下的article.article
            try:
                article = page.locator('div.text article.article')
                if await article.count() == 0:
                    await page.close()
                    return None
                # 获取所有文本内容
                content = await article.text_content()
                content = content.strip() if content else ""
                if not content:
                    await page.close()
                    return None
                await page.close()
                return {"url": url, "content": content}
            except Exception:
                await page.close()
                return None
        except Exception as e:
            return {"title": "获取失败", "url": url, "content": f"搜狐正文抓取失败: {str(e)}"}
    
    # def extract_relevant_chunks(self, query, texts, top_k=8, chunk_size=500):
    #     """
    #     将每条文本优先按句号（中英文）分割为片段，计算与query的相似度，返回最相关的top_k片段。
    #     """
    #     chunks = []
    #     for idx, text in enumerate(texts):
    #         # 先按段落分割
    #         for para in re.split(r'\n+', text):
    #             para = para.strip()
    #             if not para:
    #                 continue
    #             # 优先按句号（中英文）分割
    #             sentences = re.split(r'(?<=[。.!?！？])', para)
    #             for sent in sentences:
    #                 sent = sent.strip()
    #                 if not sent:
    #                     continue
    #                 # 若句子仍然过长，再按chunk_size截断
    #                 if len(sent) > chunk_size:
    #                     # 按chunk_size滑窗截断
    #                     for i in range(0, len(sent), chunk_size):
    #                         chunk = sent[i:i+chunk_size]
    #                         if chunk:
    #                             chunks.append((f"doc{idx+1}", chunk))
    #                 else:
    #                     chunks.append((f"doc{idx+1}", sent))
    #     if not chunks:
    #         return []
    #     # 2. 计算embedding
    #     chunk_texts = [c[1] for c in chunks]
    #     chunk_embs = self.embedding_func.embed_documents(chunk_texts)
    #     query_emb = self.embedding_func.embed_query(query)
    #     sims = cosine_similarity([query_emb], chunk_embs)[0]
    #     # 3. 取top_k相关片段
    #     top_indices = sims.argsort()[-top_k:][::-1]
    #     relevant_chunks = [f"[{chunks[i][0]}] {chunks[i][1]}" for i in top_indices]
    #     if self.config.DEBUG:
    #         print("\n最终喂给LLM的切片内容：")
    #         for i, chunk in enumerate(relevant_chunks, 1):
    #             print(f"[{i}] {chunk[:100]}...")
    #     return relevant_chunks

    async def search_with_local_priority(self, query, browser, search_page, cutoff_time=None):
        local_results = self.local_search(query, cutoff_time=cutoff_time)
        if self.config.DEBUG and local_results:
            self.debug_log("\n===== 本地向量库搜索结果 =====")
            for i, result in enumerate(local_results, 1):
                query_text = result["metadata"].get("query", "未知查询")
                timestamp = result["metadata"].get("timestamp", "未知时间")
                self.debug_log(f"本地结果 {i}:")
                self.debug_log(f"  关联查询: {query_text}")
                self.debug_log(f"  存储时间: {timestamp}")
                self.debug_log(f"  距离分数: {result['distance']:.4f}")
                self.debug_log(f"  内容摘要: {result['document'][:100]}...\n")
        web_results = await self.search_web(query, browser, search_page, cutoff_time=cutoff_time)
        if self.config.DEBUG and web_results:
            self.debug_log("\n===== 网络搜索结果 =====")
            for i, result in enumerate(web_results, 1):
                self.debug_log(f"网络结果 {i}:")
                self.debug_log(f"  标题: {result['title']}")
                self.debug_log(f"  URL: {result['url']}\n")
        local_texts = [r["document"] for r in local_results]
        web_texts = [r["content"] for r in web_results]
        all_texts = local_texts + web_texts
        # # RAG片段级筛选（已注释）
        # relevant_chunks = self.extract_relevant_chunks(query, all_texts, top_k=8, chunk_size=200)
        # combined_result = "\n".join(relevant_chunks) if relevant_chunks else "未找到相关结果"
        combined_result = "\n".join(all_texts) if all_texts else "未找到相关结果"
        return combined_result

    def generate_search_query(self, prompt):
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content":
                f"请根据下列房价预测问题，提取最简明、最核心的中文搜索关键词，要求：\n - 仅包含年份、城市、区县、街道/小区等地名信息； \n - 不包含“走势”、“预测”、“政策”等限定性或无关词汇；\n - 不输出任何解释、标点或额外语句，仅输出关键词，关键词之间用空格分隔。\n 示例：问 2025年Q1上海黄浦区小南门房价走势如何？→ 输出 上海 黄浦 小南门。\n 问题：{prompt}"}]
        )
        if Config.DEBUG:
            self.debug_log(f"检索式：{response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()

    def self_evaluate(self, prediction):
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content":
                f"请针对以下房价预测，列出所有潜在漏洞和不足，要求：\n - 只输出问题清单，不要复述预测内容；\n - 按“政策时效性”、“区域代表性”、“数据支撑”、“逻辑链条”、“风险遗漏”等分类，每类下可有多条；\n - 语言简洁明了。\n\n预测内容：\n{prediction}"}]
        )
        if Config.DEBUG:
            self.debug_log(f"[自我评估]:\n {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()

    def ask_needed_indicators(self, prompt):
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content":
                f"为了预测{prompt}，需要哪些宏观金融指标？请只列出简洁的中文指标名，用顿号分隔，例如：GDP、CPI、失业率、LPR"}]
        )
        return response.choices[0].message.content.strip()

    # def get_financial_data(self, indicators):
    #     start_date = f"{datetime.now().year}-01-01"
    #     end_date = datetime.now().strftime("%Y-%m-%d")
    #     financial_data = ""
    #     for ind in indicators.split("、"):
    #         symbol = indicator_mapping.get(ind.strip())
    #         if not symbol:
    #             financial_data += f"{ind}：暂不支持的数据\n"
    #             continue
    #         try:
    #             df = web.DataReader(symbol, "fred", start_date, end_date)
    #             latest = df.iloc[-1].values[0]
    #             financial_data += f"{ind}：{latest}\n"
    #         except Exception as e:
    #             financial_data += f"{ind}：获取失败（{str(e)}）\n"
    #     return financial_data

    async def llm_with_iteration(self, prompt):
        # 用大模型解析预测时间，得到cutoff_time（如2024Q3->2024.10.1之前，2024年3月-4月->2024.5.1之前）
        cutoff_time = None
        time_parse_prompt = (
            f"请从下列房价预测问题中提取出最晚的时间点，并将其转换为'YYYY-MM-DD'格式的下一个自然月1日。例如：'2024Q3'输出'2024-10-01'，'2024年3月-4月'输出'2024-05-01'，'2023年'输出'2024-01-01'。只输出日期，不要解释。\n问题：{prompt}"
        )
        try:
            response = self.client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": time_parse_prompt}]
            )
            cutoff_time = response.choices[0].message.content.strip()
            if not re.match(r"^20\d{2}-\d{2}-\d{2}$", cutoff_time):
                cutoff_time = None
        except Exception as e:
            if self.config.DEBUG:
                self.debug_log(f"解析预测时间失败: {e}")
            cutoff_time = None
        iterations = 0
        search_results = ""
        current_prediction = ""
        used_queries = set()
        # 检索经验库，拼接相似历史经验和经验总结
        similar_exps = self.exp_lib.retrieve_similar(prompt, top_k=2, cutoff_time=cutoff_time)
        exp_summary = self.exp_lib.summarize_experience(self.client, max_records=8)
        exp_context = ""
        if similar_exps:
            exp_context += "\n\n【历史相关经验】\n"
            for i, exp in enumerate(similar_exps, 1):
                exp_context += f"经验{i}: 问题：{exp['query']}\n预测：{exp['prediction']}\n自我评估：{exp['evaluation']}\n"
        if exp_summary:
            exp_context += f"\n【经验总结】\n{exp_summary}\n"
        if Config.DEBUG:
            self.debug_log(f"经验库上下文：{exp_context}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--enable-features=NetworkService,NetworkServiceInProcess"],
            )
            search_page = await browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
            )
            try:
                while iterations < self.config.MAX_ITERATIONS:
                    if Config.DEBUG:
                        self.debug_log(f"\n[===============第{iterations + 1}轮迭代===============]：")
                    if iterations == 0:
                        search_query = self.generate_search_query(prompt)
                        used_queries.add(search_query)
                        search_results = await self.search_with_local_priority(search_query, browser, search_page, cutoff_time=cutoff_time)
                        enhanced_prompt = (
                            f"当前时间：{datetime.now().strftime('%Y-%m-%d')}\n"
                            f"请基于下列政策和市场信息，预测“{prompt}”的房价走势。要求：\n"
                            f"1. 先用一句话给出明确预测结论（如“预计2025年Q3上海浦东严桥路房价将温和上涨，涨幅约xx-xx”, 置信度为xx）。\n"
                            f"2. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果1...”）。\n"
                            f"3. 简要列出可能的风险或不确定性因素。\n"
                            f"4. 输出内容分为“预测结论”、“主要依据”、“风险提示”三部分，禁止输出与预测无关的内容。\n\n"
                            f"{exp_context}"
                            f"搜索结果：\n{search_results}\n"
                        )
                    else:
                        assess_result = self.self_evaluate(current_prediction)
                        used_query_str = "；".join(used_queries)
                        new_query_prompt = (
                            f"请根据对“{prompt}”的预测自我评估结果：{assess_result}，明确指出还需补充哪些关键信息或数据（如政策时效、区域供需、最新市场数据、具体楼盘信息等），并将这些补充点转化为最简明、最核心的中文检索关键词。要求：仅输出关键词，关键词之间用空格分隔，不输出任何解释或多余语句。关键词的数量不要超过5个。\n"
                            f"请不要重复使用以下已用过的关键词组合：{used_query_str}"
                        )
                        new_query = self.generate_search_query(new_query_prompt)
                        used_queries.add(new_query)
                        new_search = await self.search_with_local_priority(new_query,
                                                                        browser,
                                                                        search_page, cutoff_time=cutoff_time)
                        search_results += "\n" + new_search
                        enhanced_prompt = (
                            f"基于初始预测：{current_prediction}\n"
                            f"和补充搜索到的信息：\n{new_search}\n"
                            f"{exp_context}"
                            f"重新预测问题“{prompt}”, 给出更准确的预测。要求：\n"
                            f"1. 先用一句话给出明确预测结论（如“预计2025年Q3上海浦东严桥路房价将温和上涨，涨幅约xx-xx”, 置信度为xx）。\n"
                            f"2. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果...”）。\n"
                            f"3. 简要列出可能的风险或不确定性因素。\n"
                            f"4. 输出内容分为“预测结论”、“主要依据”、“风险提示”三部分，禁止输出与预测无关的内容。\n\n"
                        )
                        if iterations == self.config.MAX_ITERATIONS - 1:
                            enhanced_prompt = (
                                f"基于初始预测：{current_prediction}\n"
                                f"和补充搜索到的信息：\n{new_search}\n"
                                f"{exp_context}"
                                f"对问题“{prompt}”, 进行最终预测。要求：\n"
                                f"1. 用一句话给出明确预测结论（如“预计2025年Q3上海浦东严桥路房价将温和上涨，涨幅约xx-xx”, 置信度为xx）。\n"
                                f"2. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果...”）。\n"
                                f"3. 输出内容分为“预测结论”、“主要依据”，禁止输出与预测无关的内容。\n\n"
                            )

                    response = self.client.chat.completions.create(
                        model="qwen-long",
                        messages=[{"role": "user", "content": enhanced_prompt}]
                    )
                    current_prediction = response.choices[0].message.content.strip()
                    if self.config.DEBUG and iterations < self.config.MAX_ITERATIONS - 1:
                        self.debug_log(f"\n[第{iterations + 1}轮预测]：{current_prediction}")
                    iterations += 1
            finally:
                await search_page.close()
                await browser.close()
        # 存储本次经验
        self.exp_lib.add(
            query=prompt,
            keywords="；".join(list(used_queries)),
            prediction=current_prediction,
            evaluation=self.self_evaluate(current_prediction)
        )
        return current_prediction

if __name__ == "__main__":
    query = "2025年Q1南京东路房价走势如何？"
    agent = Agent(Config())
    original_stdout = sys.stdout
    async def main():
        # 清空log.md
        with open('log.md', 'w', encoding='utf-8') as logf:
            logf.write('')
        with open('answer.md', 'w', encoding='utf-8') as f:
            sys.stdout = f
            result = await agent.llm_with_iteration(query)
            print("\n* 最终预测结果:\n * " + result)
        sys.stdout = original_stdout
        print("\n✅ 预测结果已写入文件 answer.md")
        if Config.DEBUG:
            print("\n✅ DEBUG日志已写入文件 log.md")
    asyncio.run(main())

    # TODO:
    # 用之前的数据测试模型效果(比如xx年Qx出了一个大政策,房价改变很大,预测之后一段时间的走势来做验证)
    # 大模型做网络搜索



