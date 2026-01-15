import os
from llm_utils import call_llm
from config import Config
import json
from prompts import PROMPTS, SYSTEM_PROMPTS
from datetime import datetime
import re

from agents.query_agent import QueryAgent
from agents.search_agent import SearchAgent
from agents.predictor_agent import PredictorAgent
from agents.reflector_agent import ReflectorAgent
from agents.memory_agent import MemoryAgent
from agents.evaluator_agent import EvaluatorAgent

class HousePriceAgent:
    def __init__(self, config: Config):
        self.config = config
        self.region = None  # 解析出的区域
        self.time_range = None  # 解析出的时间范围（如"2024年上半年"）
        self.search_history = []  # 搜索记录
        self.trajectory = []  # 记录轨迹
        # 使用配置中定义的 COT 路径
        self.cot_file = getattr(config, 'COT_TRAJECTORY_PATH', 'cot_trajectory.json')
        self.answer_path = config.ANSWER_PATH  # 最终答案存储路径
        self.current_cot = None
        self.memory_agent = MemoryAgent(config)
        self.reflections = "" # 动态加载

        # 初始化子 Agent
        self.query_agent = QueryAgent(config)
        self.search_agent = SearchAgent(config)
        self.predictor_agent = PredictorAgent(config)
        self.reflector_agent = ReflectorAgent(config)
        self.evaluator_agent = EvaluatorAgent(config)

        self.cot_field_mapping = {
            "policy": "政策分析",
            "regional": "区域特性",
            "volume_price": "量价关系",
            "time_boundary": "时间边界",
            "historical_reflection": "历史反思复用",
        }

    def save_answer(self, question: str, prediction: str, actual: str, score: int):
        """追加保存最终答案到文件（answer.json为数组）"""
        answer = {
            "question": question,
            "prediction": prediction,
            "actual": actual,
            "score": score if score != -1 else "null"
        }
        answers = []
        filename = self.answer_path + f"{self.region.strip()}.json"
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    answers = json.load(f)
                if not isinstance(answers, list):
                    answers = [answers]
            except Exception:
                answers = []
        answers.append(answer)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)

    def record_trajectory(self, step: str, content: str, cot: str = ""):
        """
        记录轨迹步骤到当前COT对象
        step为步骤名称（如解析、搜索、预测），content为具体内容
        """
        entry = f"[{step}] {content}"
        self.trajectory.append(entry)
        
        # 仅记录step和content到步骤列表
        step_data = {
            "step": step,
            "content": content
        }
        if self.current_cot is not None:
            self.current_cot["steps"].append(step_data)

        # 如果有COT内容，单独记录到步骤中
        if cot:
            self.trajectory.append(f"[{step} COT] {cot}")
            if self.current_cot is not None:
                self.current_cot["steps"].append({
                "step": f"{step} COT",
                "content": cot
            })
    
    def _save_current_cot(self):
        """保存当前COT对象到文件"""
        if not self.current_cot:
            return
        # 读取已有的 COT 列表（如果存在），确保始终将当前 COT 追加
        cots = []
        if os.path.exists(self.cot_file):
            try:
                with open(self.cot_file, "r", encoding="utf-8") as f:
                    cots = json.load(f)
                if not isinstance(cots, list):
                    cots = []
            except Exception:
                cots = []

        # 标注时间戳并追加当前 COT
        self.current_cot["timestamp"] = datetime.now().isoformat()
        cots.append(self.current_cot)

        # 仅保留最近 5 条，防止文件过长
        if len(cots) > 5:
            cots = cots[-5:]

        # 写回文件
        with open(self.cot_file, "w", encoding="utf-8") as f:
            json.dump(cots, f, ensure_ascii=False, indent=2)

    def extract_current_features(self, region: str, time_range: str, info: str = None) -> list:
        """根据区域和时间范围，以及可选的info文本，抽取一组用于检索/归类的标签（tags）。

        该方法使用启发式规则生成标签，目的是后续检索历史反思时进行快速相似度匹配。
        返回tags为字符串列表，例如：['核心城区','限购政策','量价背离']
        """
        tags = []
        if not region:
            return tags

        r = region.lower()
        # 区域类型检测（示例规则，可扩展）
        core_indicators = ['徐汇', '静安', '黄浦', '虹口']
        if any(kw.lower() in r for kw in core_indicators):
            tags.append('核心城区')
        else:
            tags.append('非核心城区')

        # 水域/滨江等地理特征
        if any(x in region for x in ['滨江', '江', '码头']):
            tags.append('滨江/沿河')

        # 时间相关的政策敏感期检测
        if any(x in time_range for x in ['下半年', '上半年', '今年', '明年']):
            tags.append('短期预测')

        # 从 info 中尝试抽取量价或政策关键词（若提供）
        if info:
            info_lower = info.lower()
            if '限购' in info_lower or '限售' in info_lower or '调控' in info_lower:
                tags.append('限购/调控')
            if '成交量' in info_lower or '成交' in info_lower:
                tags.append('成交量信号')
            if '价格' in info_lower or '涨' in info_lower or '降' in info_lower:
                tags.append('价格信号')

        # 去重并返回
        unique = []
        for t in tags:
            if t not in unique:
                unique.append(t)
        return unique

    def load_persistent_memory(self, current_tags: list = None) -> str:
        """
        利用 MemoryAgent 从向量数据库中检索相关的历史反思
        """
        # 可以结合当前区域和时间范围信息进行关联搜索
        query_text = f"{self.region} {self.time_range}"
        return self.memory_agent.search_reflections(query_text=query_text, tags=current_tags)


    def load_recent_cot(self, n=3) -> str:
        """读取最近n条COT记录（新格式），返回字符串摘要"""
        if os.path.exists(self.cot_file):
            try:
                with open(self.cot_file, "r", encoding="utf-8") as f:
                    cots = json.load(f)
                if not isinstance(cots, list):
                    return "COT格式错误"
                
                # 取最近n条
                recent = cots[-n:] if len(cots) >= n else cots
                summary = []
                for cot in recent:
                    # 优先显示COT专用步骤（政策分析、区域特性等）
                    cot_steps = [s for s in cot.get("steps", []) if s["step"] in self.cot_field_mapping.values()]
                    # 最多显示5个核心步骤
                    steps = "\n  ".join([f"{s['step']}: {s['content'][:50]}..." for s in cot_steps[:5]])
                    accurate_tag = cot.get('accurate', 'unknown')
                    summary.append(
                        f"查询: {cot.get('query', '未知')}\n"
                        f"预测结果: {cot.get('predict_trend', '未知')} | accurate: {accurate_tag}\n"
                        f"核心分析步骤:\n  {steps}\n"
                    )
                return "\n".join(summary)
            except Exception as e:
                return f"读取COT失败: {str(e)}"
        return "无COT记录"

    def load_recent_cot_block(self, n=3) -> str:
        """返回最近n条COT的结构化块，包含 timestamp 与 accurate 标签，便于放入 prompt 中"""
        if os.path.exists(self.cot_file):
            try:
                with open(self.cot_file, "r", encoding="utf-8") as f:
                    cots = json.load(f)
                if not isinstance(cots, list):
                    return ""

                recent = cots[-n:] if len(cots) >= n else cots
                blocks = []
                for cot in recent:
                    acc = cot.get('accurate', 'unknown')
                    qry = cot.get('query', '未知')
                    pred = cot.get('predict_trend', '未知')
                    is_acc = "预测准确" if acc == "true" else ("预测不准确" if acc == "false" else "")
                    blocks.append(f"query={qry} | pred={pred}" + f" | {is_acc}" if is_acc else "")
                return "\n".join(blocks)
            except Exception:
                return ""
        return ""

    async def parse_query(self, query: str) -> tuple:
        """从用户问题中解析区域和时间范围"""
        # 初始化当前COT对象
        self.current_cot = {
            "query": query,
            "predict_trend": None,
            "steps": [],
            "accurate": "unknown"
        }
        
        region, time_range = await self.query_agent.parse(query)
        self.region = region
        self.time_range = time_range

        return region, time_range
    
    async def search_related_info(self, region: str, time_range: str) -> str:
        """联网搜索影响房价的政策、新闻等信息，并记录COT"""
        output = await self.search_agent.search(region, time_range, self.reflections)
        self.search_history.append(f"搜索信息（{time_range}）：{output}")
        return output

    async def predict_trend(self, region: str, time_range: str, info: str) -> str:
        """基于搜索信息预测房价趋势（上升/下降/持平）及幅度，记录COT"""
        # 根据配置决定是否在 prompt 中包含历史反思和最近COT
        reflection_history = ""
        recent_cot = ""
        mode = getattr(self.config, 'HISTORY_MODE', 'ENABLE_BOTH')
        # 先抽取当前特征标签，用于基于标签的反思检索
        current_tags = self.extract_current_features(region, time_range, info)
        if mode in ('ENABLE_REFLECTION', 'ENABLE_BOTH'):
            reflection_history = self.load_persistent_memory(current_tags)
        if mode in ('ENABLE_COT', 'ENABLE_BOTH'):
            recent_cot = self.load_recent_cot_block(3)

        output = await self.predictor_agent.predict(
            region, time_range, info, reflection_history, recent_cot
        )

        # 尝试解析结构化 COT 并做基本验证/归一化
        if "房价预测结果:" in output:
            cot_raw, pred = output.split("房价预测结果:", 1)
            pred = pred.strip()
            cot_raw = cot_raw.strip()
        else:
            # 若模型未严格输出模板，则把全部作为COT，预测结果取末尾行
            lines = output.splitlines()
            pred = lines[-1].strip() if lines else ""
            cot_raw = "\n".join(lines[:-1]).strip()

        cot_struct = self._normalize_cot(cot_raw)

        # 拆分COT字段为独立步骤
        for field, step_name in self.cot_field_mapping.items():
            content = cot_struct.get(field, "").strip()
            if content:  # 只记录有内容的步骤
                self.record_trajectory(step_name, content)
        
        # 设置预测结果并保存当前COT
        self.current_cot["predict_trend"] = pred
        # 如果未启用进化，则保存当前COT
        # 如果启用了,要等获取实际趋势后赋值accurate再保存
        if not Config.ENABLE_EVOLUTION:
            self._save_current_cot()
        
        return pred

    def _normalize_cot(self, cot_text: str) -> dict:
        """把自由文本的COT尝试解析为固定字段的字典；不足字段用空字符串填充。

        目标字段：policy, regional, volume_price, time_boundary, historical_reflection
        对于包含量化数据的段落，保留原文以便审计。
        """
        fields = {
            "policy": "",
            "regional": "",
            "volume_price": "",
            "time_boundary": "",
            "historical_reflection": "",
            "conclusion": "",
            "raw": cot_text
        }
        if not cot_text:
            return fields

        # 优先使用正则在多行文本中精确抽取以数字序号(1.-5.)开头的段落
        try:
            pattern = re.compile(r'(?ms)^\s*([1-5])\.\s*(.*?)(?=^\s*[1-5]\.\s*|\Z)')
            matches = pattern.findall(cot_text)
            if matches:
                key_map = { '1': 'policy', '2': 'regional', '3': 'volume_price', '4': 'time_boundary', '5': 'historical_reflection', '6': 'conclusion' }
                for idx, content in matches:
                    key = key_map.get(idx)
                    if not key:
                        continue
                    seg = content.strip()
                    # 如果段落以小节标题开头（如 "政策分析：..."），去掉标题部分
                    seg = re.sub(r'^[^\n:：]{0,30}[：:\s]+', '', seg, count=1).strip()
                    fields[key] = seg
                return fields
        except Exception:
            # 若正则解析失败，再走回退逻辑
            pass

        # 回退：按关键词或小节标题抽取（兼容未编号或使用标题的情况）
        keywords = {
            "policy": ["政策分析", "政策"],
            "regional": ["区域特性", "供应/需求", "历史价格"],
            "volume_price": ["量价关系", "量价", "成交量", "以价换量"],
            "time_boundary": ["时间边界", "时滞", "滞后"],
            "historical_reflection": ["历史反思", "参考案例", "复用"],
            "conclusion": ["结论", "预测结果"]
        }
        for key, kws in keywords.items():
            for kw in kws:
                pos = cot_text.find(kw)
                if pos != -1:
                    seg = cot_text[pos:]
                    fields[key] = seg.split('\n\n', 1)[0].strip()
                    break

        return fields

    async def get_actual_trend(self, region: str, time_range: str) -> str:
        """联网搜索实际房价趋势（上升/下降/持平）及幅度（支持范围）"""
        return await self.search_agent.get_actual(region, time_range)

    # 修改agent.py的generate_reflection方法
    async def generate_reflection(self, query: str, prediction: str,
                                actual: str, score: int, info: str):
        current_trajectory = "\n".join(self.trajectory)
        # 从当前上下文抽取标签并基于标签检索历史反思摘要
        current_tags = self.extract_current_features(getattr(self, 'region', None), getattr(self, 'time_range', None), info)
        persistent_memory = self.load_persistent_memory(current_tags)
        recent_cot = self.load_recent_cot(3)

        same_reflections = self.get_same_query_reflections(query)
        history_reminder = ""
        if same_reflections:
            # 提取历史反思的核心要点
            history_points = []
            for idx, r in enumerate(same_reflections, 1):
                reflection_text = r.get('reflection_text', '')
                # 提取关键部分（错误分析/改进策略等）
                if "反思（成功）" in reflection_text:
                    parts = reflection_text.split('\n')
                    key_points = [p for p in parts if any(k in p for k in ["关键因素", "推理逻辑", "可复用框架"])]
                else:
                    parts = reflection_text.split('\n')
                    key_points = [p for p in parts if any(k in p for k in ["错误分析", "失败原因", "改进策略"])]
                history_points.append(f"第{idx}次反思要点：\n" + "\n".join(key_points[:3]))  # 取前3个要点
            history_reminder = (
                "\n注意：该问题已有历史反思记录，新反思需避免重复，补充新视角或深化分析。\n"
                + "\n".join(history_points)
            )

        if score >= Config.SCORE_THRESHOLD:
            # current COT 的 accurate 字段设置为 true
            if self.current_cot is not None:
                self.current_cot['accurate'] = 'true'
        else:
            if self.current_cot is not None:
                self.current_cot['accurate'] = 'false'

        self._save_current_cot()
        
        reflection = await self.reflector_agent.reflect(
            query, prediction, actual, score, info, history_reminder,
            current_trajectory, recent_cot, persistent_memory
        )

        # 保存为 JSON，便于检索和分析
        entry = {
            "type": "success" if score >= Config.SCORE_THRESHOLD else "failure",
            "query": query,
            "region": getattr(self, 'region', None),
            "time_range": getattr(self, 'time_range', None),
            "tags": current_tags,
            "prediction": prediction,
            "actual": actual,
            "score": score,
            "reflection_text": reflection,
            "timestamp": datetime.now().isoformat()
        }
        path = getattr(self.config, 'REFLECTION_HISTORY_PATH', 'reflection_history.json')
        reflections = []
        # 获取历史反思记录
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reflections = json.load(f)
                if not isinstance(reflections, list):
                    reflections = []
            except Exception:
                reflections = []
        # 添加当前反思记录并重新写入
        reflections.append(entry)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(reflections, f, ensure_ascii=False, indent=2)
            
        # 同时保存到向量数据库
        self.memory_agent.add_reflection(entry)
        
        return reflection

    async def check_experience(self, region: str) -> bool:
        """检查向量数据库中是否有该区域或相关经验"""
        return self.memory_agent.has_experience(region)

    async def get_relevant_reflections(self, region: str) -> str:
        """获取相关的历史反思记录（从向量数据库）"""
        return self.memory_agent.search_reflections(query_text=region, n_results=3)

    async def predict(self, region: str, time_range: str, reflections: str = "") -> tuple:
        """核心预测逻辑：搜索信息 -> 解析反思 -> 生成预测内容。"""
        if reflections:
             self.reflections = reflections
        
        search_info = await self.search_related_info(region, time_range)
        prediction_trend = await self.predict_trend(region, time_range, search_info)
        return prediction_trend, search_info

    def get_same_query_reflections(self, query: str) -> list:
        """获取历史中与当前问题相同的反思记录"""
        path = getattr(self.config, 'REFLECTION_HISTORY_PATH', 'reflection_history.json')
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reflections = json.load(f)
            if not isinstance(reflections, list):
                return []

            same_queries = []
            for r in reflections:
                if r.get('query') != query:
                    continue
                if getattr(self, 'region', None) and r.get('region') and r.get('region') != getattr(self, 'region'):
                    continue
                same_queries.append(r)
            return same_queries[-3:]
        except Exception:
            return []