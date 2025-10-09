from dashscope import Generation
from config import Config


class HousePriceAgent:
    def __init__(self, config: Config):
        self.config = config
        self.region = None  # 解析出的区域
        self.time_range = None  # 解析出的时间范围（如"2024年上半年"）
        self.search_history = []  # 搜索记录

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
        return region, time_range

    async def search_related_info(self, region: str, time_range: str) -> str:
        """联网搜索影响房价的政策、新闻等信息"""
        prompt = f"""
        请联网搜索{region}在{time_range}之前的房价相关政策（如限购、贷款政策）、
        经济数据（如GDP、人口流入）、市场动态（如成交量、新盘供应），总结为利好和利空因素，
        每条因素用一句话说明，最多各5条。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            enable_search=True,  # 启用联网搜索
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"搜索信息失败：{response.message}")

        info = response.output.text.strip()
        self.search_history.append(f"搜索信息（{time_range}）：{info}")
        return info

    async def predict_trend(self, region: str, time_range: str,
                            info: str) -> str:
        """基于搜索信息预测房价趋势（上升/下降/持平）"""
        prompt = f"""
        根据以下信息，预测{region}在{time_range}的房价趋势，只需输出：上升、下降或持平。
        信息：{info}
        输出格式：严格输出“上升”“下降”或“持平”中的一个，不添加任何额外内容。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"预测失败：{response.message}")

        trend = response.output.text.strip()
        # 确保输出符合格式
        if trend not in ["上升", "下降", "持平"]:
            raise ValueError(f"预测结果格式错误：{trend}")
        return trend

    async def get_actual_trend(self, region: str, time_range: str) -> tuple:
        """联网搜索实际房价趋势（上升/下降/持平）及幅度描述"""
        prompt = f"""
        请联网查询{region}在{time_range}的实际房价趋势，输出格式：
        趋势：[上升/下降/持平]
        幅度描述：[如“大幅上升”“小幅下降”“基本持平”等]

        若无法找到确切数据，趋势填“未知”，幅度描述填“无数据”。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            enable_search=True,  # 启用联网搜索
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"获取实际趋势失败：{response.message}")

        result = response.output.text.strip()
        trend = \
        [line.split("：")[1] for line in result.split("\n") if "趋势：" in line][
            0]
        amplitude = [line.split("：")[1] for line in result.split("\n") if
                     "幅度描述：" in line][0]
        return trend, amplitude

    async def generate_reflection(self, query: str, prediction: str,
                                  actual: str, score: int, info: str):
        """生成反思报告（分析偏差原因和改进建议）"""
        prompt = f"""
        房价预测反思：
        问题：{query}
        预测趋势：{prediction}
        实际趋势：{actual}
        评分：{score}分
        预测时使用的信息：{info}

        请分析预测与实际结果的偏差原因（如遗漏关键政策、错误判断因素权重等），
        并给出3条具体改进建议（如“需重点关注房贷利率调整”）。
        """
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"生成反思失败：{response.message}")

        reflection = response.output.text.strip()
        # 保存到历史记录
        with open(self.config.REFLECTION_HISTORY_PATH, "a",
                  encoding="utf-8") as f:
            f.write(f"## 问题：{query}\n")
            f.write(f"预测：{prediction} | 实际：{actual} | 评分：{score}\n")
            f.write(f"反思：{reflection}\n\n")
        return reflection