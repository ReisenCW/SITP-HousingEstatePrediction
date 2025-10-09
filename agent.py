import os
from dashscope import Generation
from config import Config


class HousePriceAgent:
    def __init__(self, config: Config):
        self.config = config
        self.region = None  # 解析出的区域
        self.time_range = None  # 解析出的时间范围（如"2024年上半年"）
        self.search_history = []  # 搜索记录
        self.trajectory = []  # 记录轨迹
        self.cot_file = "cot_trajectory.md"  # COT持久化文件

    def record_trajectory(self, step: str, content: str, cot: str = None):
        """
        记录轨迹步骤：step为步骤名称（如解析、搜索、预测），content为具体内容，cot为思维链（可选）
        COT内容会单独保存到cot_trajectory.md
        """
        entry = f"[{step}] {content}"
        self.trajectory.append(entry)
        if cot:
            cot_entry = f"[{step} COT] {cot}"
            self.trajectory.append(cot_entry)
            # 追加写入COT持久化文件
            with open(self.cot_file, "a", encoding="utf-8") as f:
                f.write(cot_entry + "\n")

    def load_persistent_memory(self) -> str:
        """读取历史反思记录作为持久记忆"""
        if os.path.exists(self.config.REFLECTION_HISTORY_PATH):
            with open(self.config.REFLECTION_HISTORY_PATH, "r", encoding="utf-8") as f:
                return f.read()
        return "无历史反思记录"

    def load_recent_cot(self, n=3) -> str:
        """读取最近n条COT思维链"""
        if os.path.exists(self.cot_file):
            with open(self.cot_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return "\n".join(lines[-n:]) if lines else "无COT记录"
        return "无COT记录"

    async def parse_query(self, query: str) -> tuple:
        """从用户问题中解析区域和时间范围"""
        prompt = f"""
        请从问题中提取房价预测的区域和时间范围，输出格式：
        区域：[具体区域，如北京市朝阳区]
        时间范围：[预测的时间段，如2024年下半年]

        问题：{query}
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"解析问题失败：{response.message}")

        # 提取区域和时间范围
        result = response.output.text.strip()
        region = \
        [line.split("：")[1] for line in result.split("\n") if "区域：" in line][
            0]
        time_range = [line.split("：")[1] for line in result.split("\n") if
                      "时间范围：" in line][0]
        self.region = region
        self.time_range = time_range

        self.record_trajectory("解析问题", f"原始查询：{query}；提取区域：{region}，时间范围：{time_range}")
        return region, time_range

    async def search_related_info(self, region: str, time_range: str) -> str:
        """联网搜索影响房价的政策、新闻等信息，并记录COT"""
        prompt = f"""
        请联网搜索与{region}房价相关且发生在{time_range}及其之前6个月内的相关消息,新闻或政策，总结为利好和利空因素，注意时间越久远，相关性越低，因此时间距离较大时只需要考虑那些影响房价较大的因素。
        每条消息简明扼要，注明时间，来源和标题，用一句话说明，只输出消息列表，不要解释。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            enable_search=True,  # 启用联网搜索
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"搜索信息失败：{response.message}")

        output = response.output.text.strip()
        # 拆分消息和COT
        if "COT:" in output:
            info, cot = output.split("COT:", 1)
            info = info.strip()
            cot = cot.strip()
        else:
            info, cot = output, ""
        self.search_history.append(f"搜索信息（{time_range}）：{info}")
        self.record_trajectory(f"搜索", f"区域：{region}，时间范围：{time_range}；结果：{info}", cot)
        return info

    async def predict_trend(self, region: str, time_range: str, info: str) -> str:
        """基于搜索信息预测房价趋势（上升/下降/持平），并记录COT"""
        prompt = f"""
        综合权衡以下提供的所有利好与利空信息，客观预测{region}在{time_range}的房价趋势，只需输出：上升、下降或持平。
        信息：{info}
        输出格式：幅度,趋势, 如“大幅,上升”“小幅,下降”“基本,持平”
        并请给出你的推理过程（COT），说明你如何权衡各因素得出结论。
        输出格式：预测结果\nCOT: ...
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        output = response.output.text.strip()
        if "COT:" in output:
            pred, cot = output.split("COT:", 1)
            pred = pred.strip()
            cot = cot.strip()
        else:
            pred, cot = output, ""
        self.record_trajectory("预测", f"区域：{region}，时间范围：{time_range}；预测结果：{pred}", cot)
        return pred

    async def get_actual_trend(self, region: str, time_range: str) -> str:
        """联网搜索实际房价趋势（上升/下降/持平）及幅度描述"""
        prompt = f"""
        请联网查询{region}在{time_range}的实际房价趋势，
        输出格式：幅度,趋势, 如“大幅,上升”“小幅,下降”“基本,持平”
        不添加任何额外内容
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            enable_search=True,  # 启用联网搜索
            result_format="text"
        )
        return response.output.text.strip()

    # 修改agent.py的generate_reflection方法
    async def generate_reflection(self, query: str, prediction: str,
                                actual: str, score: int, info: str):
        # 1. 定义稀疏奖励信号
        res = "成功" if score == 80 else "有偏差" if score == 40 else "失败"
        # 2. 获取当前轨迹（拼接为字符串）
        current_trajectory = "\n".join(self.trajectory)
        # 3. 获取持久记忆（历史反思）
        persistent_memory = self.load_persistent_memory()
        # 4. 获取最近3条COT
        recent_cot = self.load_recent_cot(3)

        # 2. 构建提示词，结构化输出
        prompt = f"""
        你是房价预测智能体的反思模块（reflect_llm），请基于以下信息进行分析：
        问题：{query}
        当前推理轨迹：
        {current_trajectory}
        最近3条思维链（COT）：
        {recent_cot}
        历史反思记录：
        {persistent_memory}
        预测结果：{prediction}
        实际结果：{actual}
        评分：{score}分（{res}）
        预测时使用的信息：{info}

        输出格式：
        1. 错误分析：指出之前推理中的错误或不足
        2. 失败原因：总结导致预测失败/偏差的核心原因
        3. 改进策略：提出3条具体可操作的改进建议
        要求：内容简明扼要，建议可直接用于未来预测。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        reflection = response.output.text.strip()
        # 保存到长久记忆（同之前逻辑）
        with open(self.config.REFLECTION_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(f"## 问题：{query}\n")
            f.write(f"预测：{prediction} | 实际：{actual} | 评分：{score}\n")
            f.write(f"轨迹摘要：{current_trajectory[:200]}...\n")
            f.write(f"最近COT：{recent_cot}\n")
            f.write(f"反思：{reflection}\n")
        return reflection