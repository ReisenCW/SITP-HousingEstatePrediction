import os
from dashscope import Generation
from config import Config
import json
from prompts import PROMPTS
from datetime import datetime
import re

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
        self.reflections = self.load_persistent_memory()

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
            "score": score if score is not None else "null"
        }
        answers = []
        if os.path.exists(self.answer_path):
            try:
                with open(self.answer_path, "r", encoding="utf-8") as f:
                    answers = json.load(f)
                if not isinstance(answers, list):
                    answers = [answers]
            except Exception:
                answers = []
        answers.append(answer)
        with open(self.answer_path, "w", encoding="utf-8") as f:
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
        
        cots = []
        if os.path.exists(self.cot_file):
            try:
                with open(self.cot_file, "r", encoding="utf-8") as f:
                    cots = json.load(f)
                if not isinstance(cots, list):
                    cots = []
            except Exception:
                cots = []
        
            self.current_cot["timestamp"] = datetime.now().isoformat()
            cots.append(self.current_cot)
        
        with open(self.cot_file, "w", encoding="utf-8") as f:
            json.dump(cots, f, ensure_ascii=False, indent=2)

    def load_persistent_memory(self) -> str:
        """读取历史反思记录作为持久记忆。

        优先解析 JSONL（每行一个 JSON 对象）并返回最近若干条的摘要，供 prompt 使用；
        若文件不是 JSONL，则返回原始文本（兼容旧格式）。
        """
        path = getattr(self.config, 'REFLECTION_HISTORY_PATH', 'reflection_history.json')
        if not os.path.exists(path):
            return "无历史反思记录"

        # 尝试按 JSONL 解析，每行一个 JSON 对象
        try:
            summaries = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        q = obj.get('query', '')
                        s = obj.get('score', '')
                        r = obj.get('reflection_text', '')
                        summaries.append(f"问题：{q} | 分数：{s} | 反思：{r[:200]}")
                    except Exception:
                        # 发现非 JSON 行，退回到原始文本读取
                        summaries = None
                        break
            if summaries is not None and summaries:
                return "\n".join(summaries[-5:])
        except Exception:
            pass

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
        
        prompt = PROMPTS["parse_query"].format(query=query)
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        if response.status_code != 200:
            raise RuntimeError(f"解析问题失败：{response.message}")

        # 提取区域和时间范围
        result = response.output.text.strip()
        region = [line.split("：")[1] for line in result.split("\n") if "区域：" in line][0]
        time_range = [line.split("：")[1] for line in result.split("\n") if "时间范围：" in line][0]
        self.region = region
        self.time_range = time_range

        return region, time_range
    
    async def search_related_info(self, region: str, time_range: str) -> str:
        """联网搜索影响房价的政策、新闻等信息，并记录COT"""
        prompt = PROMPTS["search_related_info"].format(
            region=region, 
            time_range=time_range
        )
        adjust_prompt = f"请根据先前的反思结果{self.reflections}, 修改原先的搜索prompt, 调整搜索策略, 避免犯同样的错误。原prompt: {prompt}"
        prompt = Generation.call(
            model=self.config.MODEL,
            prompt=adjust_prompt,
            enable_search=False,
            result_format="text"
        ).output.text.strip()

        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            enable_search=True,  # 启用联网搜索
            result_format="text"
        )
        output = response.output.text.strip()
        
        self.search_history.append(f"搜索信息（{time_range}）：{output}")
        return output

    async def predict_trend(self, region: str, time_range: str, info: str) -> str:
        """基于搜索信息预测房价趋势（上升/下降/持平）及幅度，记录COT"""
        # 根据配置决定是否在 prompt 中包含历史反思和最近COT
        reflection_history = ""
        recent_cot = ""
        mode = getattr(self.config, 'HISTORY_MODE', 'ENABLE_BOTH')
        if mode in ('ENABLE_REFLECTION', 'ENABLE_BOTH'):
            reflection_history = self.load_persistent_memory()
        if mode in ('ENABLE_COT', 'ENABLE_BOTH'):
            recent_cot = self.load_recent_cot_block(3)

        # 强制结构化 COT 模板，避免跳跃性推理并要求量化锚定
        prompt = PROMPTS["predict_trend"].format(
            region=region,
            time_range=time_range,
            info=info,
            reflection_history=f"历史反思记录：\n{reflection_history}" if reflection_history else "",
            recent_cot=f"最近3条思维链（COT）记录：\n{recent_cot}" if recent_cot else ""
        )
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        output = response.output.text.strip()
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
        prompt = PROMPTS["get_actual_trend"].format(region=region, time_range=time_range)
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
        current_trajectory = "\n".join(self.trajectory)
        persistent_memory = self.load_persistent_memory()
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
            prompt = PROMPTS["generate_reflection_success"].format(
                score=score,
                history_reminder=history_reminder,
                query=query,
                current_trajectory=current_trajectory,
                recent_cot=recent_cot,
                persistent_memory=persistent_memory,
                prediction=prediction,
                actual=actual,
                info=info
            )
            # current COT 的 accurate 字段设置为 true
            if self.current_cot is not None:
                self.current_cot['accurate'] = 'true'
        else:
            prompt = PROMPTS["generate_reflection_failure"].format(
                score=score,
                history_reminder=history_reminder,
                query=query,
                current_trajectory=current_trajectory,
                recent_cot=recent_cot,
                persistent_memory=persistent_memory,
                prediction=prediction,
                actual=actual,
                info=info
            )
            if self.current_cot is not None:
                self.current_cot['accurate'] = 'false'

        self._save_current_cot()
        response = Generation.call(
            model=self.config.MODEL,
            prompt=prompt,
            result_format="text"
        )
        reflection = response.output.text.strip()
        # 保存为 JSON，便于检索和分析
        entry = {
            "type": "success" if score >= Config.SCORE_THRESHOLD else "failure",
            "query": query,
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
        return reflection

    # 检索相同问题的历史反思
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
            # 筛选相同query的记录，取最近3条
            same_queries = [r for r in reflections if r.get('query') == query]
            return same_queries[-3:]  # 只取最近3条避免信息过载
        except Exception:
            return []