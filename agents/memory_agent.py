import chromadb
from chromadb.config import Settings
import os
import json
import uuid
from typing import List, Dict, Any

class MemoryAgent:
    def __init__(self, config):
        self.config = config
        self.db_path = getattr(config, 'CHROMA_DB_PATH', './chroma_db')
        
        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 获取或创建 reflection 集合
        self.collection = self.client.get_or_create_collection(
            name="reflections",
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度
        )

    def add_reflection(self, reflection_entry: Dict[str, Any]):
        """
        将反思记录添加到向量数据库
        """
        # 准备元数据和文本内容
        # 因为 Chroma 元数据不支持 list，所以将 tags 转为逗号分隔的字符串
        metadata = {
            "type": reflection_entry.get("type", "unknown"),
            "region": reflection_entry.get("region", "unknown"),
            "score": float(reflection_entry.get("score", 0)),
            "time_range": reflection_entry.get("time_range", "unknown"),
            "tags": ",".join(reflection_entry.get("tags", [])),
            "query": reflection_entry.get("query", "")
        }
        
        # 将整个反思内容作为文本进行嵌入
        content = reflection_entry.get("reflection_text", "")
        if not content:
            return

        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def search_reflections(self, query_text: str = None, tags: List[str] = None, n_results: int = 3) -> str:
        """
        基于语义和标签搜索相关的反思记录
        """
        # 简单的检索逻辑：如果有 query_text（通常是当前任务描述或特征描述），做向量搜索
        # 如果有 tags，可以作为过滤条件（Chroma 支持 where 过滤，但这里先主要用语义搜索）
        
        if not query_text and not tags:
            return ""

        # 构建搜索文本：结合 tags 和 query_text
        search_content = f"{query_text or ''} {' '.join(tags or [])}"
        
        results = self.collection.query(
            query_texts=[search_content],
            n_results=n_results
        )

        summaries = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                summaries.append(
                    f"[来源区域: {meta['region']}] [评分: {meta['score']}] [标签: {meta['tags']}]\n反思内容: {doc}"
                )
        
        return "\n---\n".join(summaries) if summaries else "无匹配的历史反思"

    def has_experience(self, region: str) -> bool:
        """
        检查是否已有该区域的相关高分经验
        """
        results = self.collection.get(
            where={"region": region}
        )
        if results and results['ids']:
            # 简单判断是否有记录，进阶可以判断 score
            return len(results['ids']) > 0
        return False
