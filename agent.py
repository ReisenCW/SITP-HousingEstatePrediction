import sys
import asyncio
from datetime import datetime
from openai import OpenAI
from config import Config

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

    def search_region_news_with_llm(self, prompt, is_first=True, evaluate_result=None, seen_msgs_str=None):
        """
        利用LLM联网搜索能力，针对问题中的区域和时间，抓取房价利好和利空消息。
        返回：{"region": 区域, "bullish": [消息...], "bearish": [消息...]}
        """
        # 1. 用LLM抽取区域关键词
        extract_region_prompt = (
            f"请从下列房价预测问题中，提取最具体的地理区域（如城市、区县、街道、小区等），只输出区域名，不要输出任何解释或多余内容。\n问题：{prompt}"
        )
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content": extract_region_prompt}],
            extra_body={"enable_search": False}
        )
        region = response.choices[0].message.content.strip().replace("\n", " ").replace("，", " ").split()[0]
        # 2. 用LLM抽取时间范围
        extract_time_prompt = (
            f"请从下列房价预测问题中，提取最相关的时间范围（如年份、季度、月份等），只输出时间，不要解释。如果没有时间信息，输出“当前”即可。\n问题：{prompt}"
        )
        time_resp = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content": extract_time_prompt}],
            extra_body={"enable_search": False}
        )
        time_str = time_resp.choices[0].message.content.strip().replace("\n", " ").replace("，", " ")
        if not time_str or "当前" in time_str:
            time_str = datetime.now().strftime("%Y-%m-%d")
        # 3. 构造利好/利空搜索指令，带时间
        bullish_prompt = (
            f"请联网搜索与“{region} 房价 利好消息”相关、发生在{time_str}及其前3-6个月内的新闻或政策，列举至多{self.config.top_k}条最具代表性的利好消息。"
            f"要求：\n- 优先选择{time_str}及其前3个月内的消息，若数量不足可补充前6个月内的重大消息；\n"
            f"- 时间越久远，要求其对房价影响力越大（如重大政策、市场转折等）；\n"
            f"- 每条简明扼要，注明时间、来源或标题；\n- 只输出消息列表，不要解释。\n- 如果相关度不够，可以少于{self.config.top_k}条，不要强行补足。"
        )
        bearish_prompt = (
            f"请联网搜索与“{region} 房价 利空消息”相关、发生在{time_str}及其前3-6个月内的新闻或政策，列举至多{self.config.top_k}条最具代表性的利空消息。"
            f"要求：\n- 优先选择{time_str}及其前3个月内的消息，若数量不足可补充前6个月内的重大消息；\n"
            f"- 时间越久远，要求其对房价影响力越大（如重大政策、市场转折等）；\n"
            f"- 每条简明扼要，注明时间、来源或标题；\n- 只输出消息列表，不要解释。\n- 如果相关度不够，可以少于{self.config.top_k}条，不要强行补足。"
        )
        if not is_first:
            # 如果不是第一次搜索，加入已见消息去重提示
            if seen_msgs_str:
                bullish_prompt += f"\n注意：不要返回下列已出现过的消息：\n{seen_msgs_str}\n"
                bearish_prompt += f"\n注意：不要返回下列已出现过的消息：\n{seen_msgs_str}\n"
            if evaluate_result:
                bullish_prompt += f"请根据对“{prompt}”的预测自我评估结果：{evaluate_result}，补充检索该区域房价相关的最新利好消息。\n"
                bearish_prompt += f"请根据对“{prompt}”的预测自我评估结果：{evaluate_result}，补充检索该区域房价相关的最新利空消息。\n"
            bullish_prompt += "请尽量从不同新闻来源、不同表述、不同角度、不同细分领域（如政策、市场、金融、人口、土地、开发商、学区、交通等）发掘更多未出现过的新消息。允许适当扩展关键词、时间范围或换用不同表达方式，避免与已见消息重复。\n只有在确实无法检索到任何新消息时，才回复'未搜索到补充消息'。如有任何不同来源、不同表述或细节的新内容，都应输出。\n"
            bearish_prompt += "请尽量从不同新闻来源、不同表述、不同角度、不同细分领域（如政策、市场、金融、人口、土地、开发商、学区、交通等）发掘更多未出现过的新消息。允许适当扩展关键词、时间范围或换用不同表达方式，避免与已见消息重复。\n只有在确实无法检索到任何新消息时，才回复'未搜索到补充消息'。如有任何不同来源、不同表述或细节的新内容，都应输出。\n"

        # 4. 联网抓取利好消息
        bullish_resp = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content": bullish_prompt}],
            extra_body={
                "enable_search": True,
                "forced_search": True
            }
        )
        bullish_list = [line.strip('-').strip() for line in bullish_resp.choices[0].message.content.strip().split('\n') if line.strip()]
        # 5. 联网抓取利空消息
        bearish_resp = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content": bearish_prompt}],
            extra_body={
                "enable_search": True,
                "forced_search": True
            }
        )
        bearish_list = [line.strip('-').strip() for line in bearish_resp.choices[0].message.content.strip().split('\n') if line.strip()]
        return {"region": region, "bullish": bullish_list, "bearish": bearish_list}
 

    def self_evaluate(self, prediction):
        # 自我评估，用于进一步补充搜索相关信息
        response = self.client.chat.completions.create(
            model="qwen-long",
            messages=[{"role": "user", "content":
                f"请针对以下房价预测，列出所有潜在漏洞和不足，要求：\n - 只输出问题清单，不要复述预测内容；\n - 按“政策时效性”、“区域代表性”、“数据支撑”、“逻辑链条”、“风险遗漏”等分类，每类下可有多条；\n - 语言简洁明了。\n\n预测内容：\n{prediction}"}],
            extra_body={"enable_search": False}
        )
        if Config.DEBUG:
            self.debug_log(f"[自我评估]:\n {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()

    async def llm_with_iteration(self, prompt):
        iterations = 0
        current_prediction = ""
        seen_msgs = set()  # 用于去重所有已抓取的消息
        while iterations < self.config.MAX_ITERATIONS:
            if Config.DEBUG:
                self.debug_log(f"\n[===============第{iterations + 1}轮迭代===============]：")
            if iterations == 0:
                search_results = self.search_region_news_with_llm(prompt)
                # 利好/利空合并去重
                unique_bullish = []
                unique_bearish = []
                for msg in search_results['bullish']:
                    if msg not in seen_msgs:
                        unique_bullish.append(msg)
                        seen_msgs.add(msg)
                for msg in search_results['bearish']:
                    if msg not in seen_msgs:
                        unique_bearish.append(msg)
                        seen_msgs.add(msg)
                if self.config.DEBUG:
                    self.debug_log(f"区域：{search_results['region']}\n利好消息：\n" + "\n".join(unique_bullish) + "\n利空消息：\n" + "\n".join(unique_bearish))
                region_news_str = f"【{search_results['region']}房价利好消息】\n" + "\n".join(unique_bullish) + "\n【{search_results['region']}房价利空消息】\n" + "\n".join(unique_bearish)
                enhanced_prompt = (
                    f"当前时间：{datetime.now().strftime('%Y-%m-%d')}\n"
                    f"请基于下列搜索到的相关消息，预测“{prompt}”的房价走势。要求：\n"
                    f"1. 综合权衡所有利好与利空消息，客观判断房价是上涨、下跌还是持平。不得片面强调任何一方。\n"
                    f"2. 只有在利好和利空势均力敌、没有明显主导证据时，才可以选择“持平”；如有一方证据明显占优，应优先给出“上涨”或“下跌”结论。\n"
                    f"3. 只能选择“上涨”、“下跌”或“持平”三者之一，且只输出一个方向及其幅度，不得同时给出涨跌幅。\n"
                    f"4. 预测结论必须用“上涨”、“下跌”或“持平”字样开头，并只给出一个方向的幅度和置信度。禁止出现“涨跌幅”并列或模糊表述。\n"
                    f"5. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果...”）。\n"
                    f"6. 简要列出可能的风险或不确定性因素。\n"
                    f"7. 输出内容分为“预测结论”、“主要依据”、“风险提示”三部分，禁止输出与预测无关的内容。\n\n"
                    f"搜索到的消息：{region_news_str}\n"
                )
            else:
                assess_result = self.self_evaluate(current_prediction)
                # 构建re_search_prompt，利用LLM联网搜索功能进行补充搜索，直接让LLM联网查找补充的利好/利空消息
                seen_msgs_str = "\n".join(seen_msgs)
               
                new_search = self.search_region_news_with_llm(prompt, False, assess_result, seen_msgs_str)
                if('未搜索到补充消息' in new_search):
                    if self.config.DEBUG:
                        self.debug_log(f"未搜索到补充消息，结束迭代。")
                    break
                # 只保留未出现过的新消息
                new_bullish = [msg for msg in new_search['bullish'] if msg not in seen_msgs]
                new_bearish = [msg for msg in new_search['bearish'] if msg not in seen_msgs]
                for msg in new_bullish + new_bearish:
                    seen_msgs.add(msg)
                if self.config.DEBUG:
                    self.debug_log(f"补充搜索到的消息：\n" + "\n".join(new_bullish) + "\n" + "\n".join(new_bearish)) 
                enhanced_prompt = (
                    f"基于上一轮的预测：{current_prediction}\n"
                    f"和补充搜索到的信息：\n【利好】\n" + "\n".join(new_bullish) + "\n【利空】\n" + "\n".join(new_bearish) + "\n"
                    f"重新预测问题“{prompt}”，给出更准确的预测。要求：\n"
                    f"1. 综合权衡所有补充的利好与利空消息，客观判断房价是上涨、下跌还是持平。不得片面强调任何一方。\n"
                    f"2. 只有在利好和利空势均力敌、没有明显主导证据时，才可以选择“持平”；如有一方证据明显占优，应优先给出“上涨”或“下跌”结论。\n"
                    f"3. 只能选择“上涨”、“下跌”或“持平”三者之一，且只输出一个方向及其幅度，不得同时给出涨跌幅。\n"
                    f"4. 预测结论必须用“上涨”、“下跌”或“持平”字样开头，并只给出一个方向的幅度和置信度。禁止出现“涨跌幅”并列或模糊表述。\n"
                    f"5. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果...”）。\n"
                    f"6. 简要列出可能的风险或不确定性因素。\n"
                    f"7. 输出内容分为“预测结论”、“主要依据”、“风险提示”三部分，禁止输出与预测无关的内容。\n\n"
                )
                if iterations == self.config.MAX_ITERATIONS - 1:
                    enhanced_prompt = (
                        f"基于上一轮的预测：{current_prediction}\n"
                        f"和补充搜索到的信息：\n【利好】\n" + "\n".join(new_bullish) + "\n【利空】\n" + "\n".join(new_bearish) + "\n"
                        f"对问题“{prompt}”，进行最终预测。要求：\n"
                        f"1. 综合权衡所有补充的利好与利空消息，客观判断房价是上涨、下跌还是持平。不得片面强调任何一方。\n"
                        f"2. 只有在利好和利空势均力敌、没有明显主导证据时，才可以选择“持平”；如有一方证据明显占优，应优先给出“上涨”或“下跌”结论。\n"
                        f"3. 只能选择“上涨”、“下跌”或“持平”三者之一，且只输出一个方向及其幅度，不得同时给出涨跌幅。\n"
                        f"4. 预测结论必须用“上涨”、“下跌”或“持平”字样开头，并只给出一个方向的幅度和置信度。禁止出现“涨跌幅”并列或模糊表述。\n"
                        f"5. 用要点列出主要依据，每条注明对应的搜索结果编号或简要来源（如“根据结果...”）。\n"
                        f"6. 输出内容分为“预测结论”、“主要依据”，禁止输出与预测无关的内容。\n\n"
                    )

            response = self.client.chat.completions.create(
                model="qwen-long",
                messages=[{"role": "user", "content": enhanced_prompt}],
                extra_body={"enable_search": False}
            )
            current_prediction = response.choices[0].message.content.strip()
            if self.config.DEBUG and iterations < self.config.MAX_ITERATIONS - 1:
                self.debug_log(f"\n[第{iterations + 1}轮预测]：{current_prediction}")
            iterations += 1

        return current_prediction

if __name__ == "__main__":
    query = "2025上半年徐汇滨江房价走势如何？"
    agent = Agent(Config())
    original_stdout = sys.stdout
    async def main():
        # 清空log.md
        with open('log.md', 'w', encoding='utf-8') as logf:
            logf.write('')
        # answer.md采用追加模式，并用分隔线隔开
        result = await agent.llm_with_iteration(query)
        with open('answer.md', 'a', encoding='utf-8') as f:
            f.write('\n' + '='*40 + '\n')
            f.write('* 预测问题：' + query + '\n')
            f.write("* 最终预测结果:\n * " + result + '\n')
        print("\n✅ 预测结果已追加到文件 answer.md")
        if Config.DEBUG:
            print("\n✅ DEBUG日志已写入文件 log.md")
    asyncio.run(main())

# 1.评估的时间跨度大一点，以年为单位(半年，一年)
# 2.预测的时间跨度还是以Q为单位，整合后形成年进行评估


