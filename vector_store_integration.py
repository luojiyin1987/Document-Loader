#!/usr/bin/env python3
"""
Vector Store Integration Module - 向量存储集成模块
提供与现有文档处理系统的集成接口
"""

# ===== 标准库导入 =====
from pathlib import Path
from typing import Any, Dict, List

# ===== 项目自定义模块导入 =====
from embeddings import SimpleEmbeddings
from vector_store import create_vector_store


class VectorStoreIntegration:
    """向量存储集成类"""

    def __init__(self, vector_store_file: str = "vector_store.pkl"):
        self.vector_store_file = vector_store_file
        self.vector_store = create_vector_store("memory")
        self.embedder = SimpleEmbeddings()

    def add_documents_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """从文本分割块添加文档到向量存储"""
        if not chunks:
            return []

        # 准备文本用于训练嵌入模型
        texts = [chunk["page_content"] for chunk in chunks]

        # 训练嵌入模型
        if not self.embedder.fitted:
            self.embedder.fit(texts)

        # 生成嵌入向量
        embeddings = []
        for chunk in chunks:
            embedding = self.embedder.embed_text(chunk["page_content"])
            embeddings.append(embedding)

        # 添加到向量存储
        doc_ids = self.vector_store.add_texts(
            [chunk["page_content"] for chunk in chunks],
            [chunk["metadata"] for chunk in chunks],
            embeddings,
        )

        return doc_ids

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似的文本块"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k)
        return results

    def search_by_metadata(
        self, filter_dict: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """根据元数据搜索"""
        return self.vector_store.filter_by_metadata(filter_dict, top_k)

    def save_vector_store(self) -> None:
        """保存向量存储"""
        # 使用更安全的JSON格式
        json_file = self.vector_store_file.replace(".pkl", ".json")
        self.vector_store.save_to_json(json_file)

    def load_vector_store(self) -> bool:
        """加载向量存储"""
        try:
            # 只支持JSON格式，更安全
            json_file = self.vector_store_file.replace(".pkl", ".json")
            if Path(json_file).exists():
                self.vector_store.load_from_json(json_file)
                return True
            # 如果JSON文件不存在，但存在pickle文件，提示用户转换
            elif Path(self.vector_store_file).exists():
                print(f"发现旧格式的pickle文件: {self.vector_store_file}")
                print("请先转换为JSON格式：")
                print("```python")
                print("import pickle")
                print("import json")
                print("from datetime import datetime")
                print("")
                print("# 加载旧格式")
                print("with open('vector_store.pkl', 'rb') as f:")
                print("    data = pickle.load(f)")
                print("")
                print("# 保存为新格式")
                print("with open('vector_store.json', 'w', encoding='utf-8') as f:")
                print("    json.dump(data, f, ensure_ascii=False, indent=2)")
                print("```")
                return False
            return False
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.vector_store.get_stats()

    def clear_vector_store(self) -> None:
        """清空向量存储"""
        self.vector_store.clear_all()


def create_vector_store_integration(
    vector_store_file: str = "vector_store.pkl",
) -> VectorStoreIntegration:
    """创建向量存储集成的工厂函数"""
    return VectorStoreIntegration(vector_store_file)


# 便捷函数
def add_chunks_to_vector_store(
    chunks: List[Dict[str, Any]], vector_store_file: str = "vector_store.pkl"
) -> List[str]:
    """便捷函数：将文本块添加到向量存储"""
    integration = create_vector_store_integration(vector_store_file)
    return integration.add_documents_from_chunks(chunks)


def search_vector_store(
    query: str, top_k: int = 5, vector_store_file: str = "vector_store.pkl"
) -> List[Dict[str, Any]]:
    """便捷函数：搜索向量存储"""
    integration = create_vector_store_integration(vector_store_file)
    if integration.load_vector_store():
        return integration.search_similar_chunks(query, top_k)
    else:
        print(f"向量存储文件 {vector_store_file} 不存在")
        return []
