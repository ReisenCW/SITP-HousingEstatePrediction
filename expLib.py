import json
from datetime import datetime

class ExpLib:
    def __init__(self, db_path="experience_db.json", embedding_func=None):
        self.db_path = db_path
        self.embedding_func = embedding_func
        self.experience_db = []
        self.load()

    def load(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.experience_db = json.load(f)
        except Exception:
            self.experience_db = []

    def save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.experience_db, f, ensure_ascii=False, indent=2)

    def add(self, query, keywords, summary, prediction, evaluation, timestamp=None):
        # timestamp: 预测时间（如2025-10-01），否则用当前时间
        record = {
            "query": query,
            "keywords": keywords,
            "summary": summary,
            "prediction": prediction,
            "evaluation": evaluation,
            "timestamp": timestamp if timestamp else datetime.now().isoformat()
        }
        self.experience_db.append(record)
        self.save()

    def retrieve_similar(self, query, top_k=3, cutoff_time=None):
        if not self.embedding_func or not self.experience_db:
            return []
        # 只用timestamp在cutoff_time之前的经验
        filtered_db = self.experience_db
        if cutoff_time:
            filtered_db = [item for item in self.experience_db if item.get("timestamp","")[:10] <= cutoff_time]
        if not filtered_db:
            return []
        query_emb = self.embedding_func.embed_query(query)
        texts = [item["query"] for item in filtered_db]
        doc_embs = self.embedding_func.embed_documents(texts)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([query_emb], doc_embs)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        return [filtered_db[i] for i in top_indices]

    def summarize_experience(self, llm_client, max_records=10):
        # 取最近max_records条经验，拼接后让LLM总结经验规则
        if not self.experience_db:
            return ""
        records = self.experience_db[-max_records:]
        text = "\n\n".join([
            f"问题：{r['query']}\n预测：{r['prediction']}\n自我评估：{r['evaluation']}" for r in records
        ])
        prompt = (
            "请根据以下历史预测与自我评估，总结出房价预测的高频因果链、经验规则或常见风险点，要求：\n"
            "- 只输出简短、清晰的规则，每条规则不超过30字；\n"
            "- 每条规则用阿拉伯数字加点编号（如1. 2. 3.）；\n"
            "- 不要输出多余解释、不要分段，只输出编号规则列表。\n"
            "示例：\n1. 政策出台后短期房价易波动\n2. 区域经济强则房价更稳健\n3. 信贷收紧会抑制涨幅\n"
            "\n历史数据如下：\n" + text
        )
        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()