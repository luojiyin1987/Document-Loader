#!/usr/bin/env python3
"""
Vector Store Module - 简单的向量存储实现
支持文档的持久化存储和相似度搜索
"""

import json

# ===== 标准库导入 =====
import math
import pickle  # nosec B403 - only used for saving, not loading (safe usage)
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

# ===== 第三方库导入 =====
try:
    import numpy as np
except ImportError:
    print("警告: 未找到 numpy，将使用纯 Python 实现")
    np = None  # type: ignore


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add_documents(
        self, documents: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """添加文档和向量到存储"""
        pass

    @abstractmethod
    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        pass

    @abstractmethod
    def search_by_metadata(
        self, filter_dict: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """根据元数据搜索文档"""
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> None:
        """删除文档"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        pass


class InMemoryVectorStore(VectorStore):
    """内存向量存储 - 简单但高效的实现"""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.doc_id_to_index: Dict[str, int] = {}
        self.metadata_index: Dict[str, List[Dict[str, Any]]] = {}
        self.created_at = datetime.now()

    def add_documents(
        self, documents: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """添加文档和向量到存储"""
        if len(documents) != len(embeddings):
            raise ValueError("文档数量和嵌入向量数量不匹配")

        for doc, embedding in zip(documents, embeddings):
            # 生成文档ID
            doc_id = doc.get(
                "id", f"doc_{len(self.documents)}_{hash(doc['page_content'])}"
            )
            doc["id"] = doc_id
            doc["stored_at"] = datetime.now().isoformat()

            # 存储文档和嵌入
            self.documents.append(doc)
            self.embeddings.append(embedding)

            # 建立ID索引
            self.doc_id_to_index[doc_id] = len(self.documents) - 1

            # 建立元数据索引
            self._build_metadata_index(doc, len(self.documents) - 1)

    def _build_metadata_index(self, doc: Dict[str, Any], index: int) -> None:
        """构建元数据索引"""
        metadata = doc.get("metadata", {})

        # 为元数据中的每个键值对建立索引
        for key, value in metadata.items():
            if key not in self.metadata_index:
                self.metadata_index[key] = []

            # 如果值还没有对应的索引列表，创建一个
            value_key = f"{key}:{value}"
            if value_key not in [x for x in self.metadata_index[key]]:
                self.metadata_index[key].append({"value": value, "indices": [index]})
            else:
                # 找到对应的值索引并添加当前文档索引
                for item in self.metadata_index[key]:
                    if item["value"] == value:
                        item["indices"].append(index)
                        break

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """余弦相似度搜索"""
        if not self.embeddings:
            return []

        # 计算查询向量与所有文档向量的相似度
        similarities: List[Dict[str, Any]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append(
                {"index": i, "similarity": similarity, "document": self.documents[i]}
            )

        # 按相似度排序并返回前top_k个结果
        similarities.sort(key=lambda x: float(x["similarity"]), reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0

        # 使用 numpy 如果可用
        if np is not None:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
        else:
            # 纯 Python 实现
            dot_product = sum(x * y for x, y in zip(vec1, vec2))
            norm1 = math.sqrt(sum(x * x for x in vec1))
            norm2 = math.sqrt(sum(x * x for x in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_by_metadata(
        self, filter_dict: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """根据元数据搜索文档"""
        matching_indices = set(range(len(self.documents)))

        for key, value in filter_dict.items():
            if key in self.metadata_index:
                # 找到匹配该元数据键值对的文档索引
                key_indices = set()
                for item in self.metadata_index[key]:
                    if item["value"] == value:
                        key_indices.update(item["indices"])

                # 取交集
                matching_indices.intersection_update(key_indices)
            else:
                # 如果元数据键不存在，返回空结果
                return []

        # 返回匹配的文档
        results = []
        for index in matching_indices:
            results.append({"index": index, "document": self.documents[index]})

        return results[:top_k]

    def delete(self, doc_ids: List[str]) -> None:
        """删除文档"""
        # 按索引从大到小排序，避免删除时索引错位
        indices_to_delete = []
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_index:
                indices_to_delete.append(self.doc_id_to_index[doc_id])

        indices_to_delete.sort(reverse=True)

        for index in indices_to_delete:
            # 删除文档和嵌入
            del self.documents[index]
            del self.embeddings[index]

            # 重建索引
            self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """重建索引"""
        self.doc_id_to_index.clear()
        self.metadata_index.clear()

        for i, doc in enumerate(self.documents):
            doc_id = doc["id"]
            self.doc_id_to_index[doc_id] = i
            self._build_metadata_index(doc, i)

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "unique_metadata_keys": len(self.metadata_index),
            "created_at": self.created_at.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0,
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    def _estimate_memory_usage(self) -> float:
        """估算内存使用量 (MB)"""
        # 简单的内存估算
        docs_size = len(str(self.documents)) / 1024 / 1024  # MB
        embeddings_size = (
            len(self.embeddings) * len(self.embeddings[0]) * 8 / 1024 / 1024
            if self.embeddings
            else 0
        )  # MB
        return docs_size + embeddings_size

    def save(self, file_path: str) -> None:
        """保存到文件"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "doc_id_to_index": self.doc_id_to_index,
            "metadata_index": self.metadata_index,
            "created_at": self.created_at.isoformat(),
        }

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def save_json(self, file_path: str) -> None:
        """保存到JSON文件（更安全的方式）"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "doc_id_to_index": self.doc_id_to_index,
            "metadata_index": self.metadata_index,
            "created_at": self.created_at.isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, _file_path: str) -> None:
        """从文件加载 (已弃用，请使用 load_json 方法)"""
        # 已移除不安全的 pickle.load() 方法
        # 请使用 load_json() 方法进行安全的文件加载
        raise NotImplementedError(
            "pickle.load() 方法已移除，请使用 load_json() 方法进行安全的文件加载。\n"
            "如果您有旧的 .pkl 文件，请先使用以下代码转换为 JSON 格式：\n"
            "```python\n"
            "# 转换旧格式到新格式\n"
            "vector_store.save_json('new_file.json')\n"
            "```"
        )

    def load_json(self, file_path: str) -> None:
        """从JSON文件加载（更安全的方式）"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            self.doc_id_to_index = data["doc_id_to_index"]
            self.metadata_index = data["metadata_index"]
            self.created_at = datetime.fromisoformat(data["created_at"])
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"文件格式错误: {e}")

    def clear(self) -> None:
        """清空存储"""
        self.documents.clear()
        self.embeddings.clear()
        self.doc_id_to_index.clear()
        self.metadata_index.clear()


class SimpleVectorStore:
    """简化的向量存储包装器 - 提供更简单的接口"""

    def __init__(self, storage_type: str = "memory"):
        if storage_type == "memory":
            self.storage = InMemoryVectorStore()
        else:
            raise ValueError(f"不支持的存储类型: {storage_type}")

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """添加单个文本"""
        doc = {"page_content": text, "metadata": metadata or {}}

        if embedding is None:
            # 这里需要一个嵌入生成器，暂时返回None
            raise ValueError("需要提供嵌入向量")

        self.storage.add_documents([doc], [embedding])
        return str(doc["id"])

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """批量添加文本"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append({"page_content": text, "metadata": metadata})

        if embeddings is None:
            raise ValueError("需要提供嵌入向量")

        self.storage.add_documents(documents, embeddings)
        return [str(doc["id"]) for doc in documents]

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        results = self.storage.similarity_search(query_embedding, top_k)
        return [
            {
                "document": result["document"],
                "similarity": result["similarity"],
                "id": result["document"]["id"],
            }
            for result in results
        ]

    def filter_by_metadata(
        self, filter_dict: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """根据元数据过滤"""
        results = self.storage.search_by_metadata(filter_dict, top_k)
        return [
            {"document": result["document"], "id": result["document"]["id"]}
            for result in results
        ]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        if doc_id in self.storage.doc_id_to_index:
            index = self.storage.doc_id_to_index[doc_id]
            return self.storage.documents[index]
        return None

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            self.storage.delete([doc_id])
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.storage.get_stats()

    def save_to_file(self, file_path: str) -> None:
        """保存到文件"""
        self.storage.save(file_path)

    def load_from_file(self, file_path: str) -> None:
        """从文件加载"""
        self.storage.load(file_path)

    def save_to_json(self, file_path: str) -> None:
        """保存到JSON文件（更安全的方式）"""
        self.storage.save_json(file_path)

    def load_from_json(self, file_path: str) -> None:
        """从JSON文件加载（更安全的方式）"""
        self.storage.load_json(file_path)

    def clear_all(self) -> None:
        """清空所有数据"""
        self.storage.clear()


# 工厂函数
def create_vector_store(storage_type: str = "memory") -> SimpleVectorStore:
    """创建向量存储的工厂函数"""
    return SimpleVectorStore(storage_type)


if __name__ == "__main__":
    # 简单测试
    print("=== 向量存储测试 ===")

    # 创建向量存储
    vector_store = create_vector_store("memory")

    # 模拟一些嵌入向量 (实际使用时应该用嵌入模型生成)
    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.6, 0.5, 0.4, 0.3, 0.2],
    ]

    # 添加文档
    sample_texts = [
        "Python是一种高级编程语言",
        "机器学习是人工智能的重要分支",
        "深度学习使用神经网络进行学习",
        "数据科学包含统计分析",
    ]

    sample_metadatas = [
        {"category": "programming", "difficulty": "easy"},
        {"category": "ai", "difficulty": "medium"},
        {"category": "ai", "difficulty": "hard"},
        {"category": "data", "difficulty": "medium"},
    ]

    # 添加文档到向量存储
    doc_ids = vector_store.add_texts(sample_texts, sample_metadatas, sample_embeddings)
    print(f"添加了 {len(doc_ids)} 个文档")

    # 搜索测试
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
    results = vector_store.search(query_embedding, top_k=2)

    print("\n=== 搜索结果 ===")
    for result in results:
        print(f"相似度: {result['similarity']:.3f}")
        print(f"文档: {result['document']['page_content']}")
        print(f"元数据: {result['document']['metadata']}")
        print(f"ID: {result['id']}")
        print()

    # 元数据过滤测试
    print("=== 元数据过滤 ===")
    filtered_results = vector_store.filter_by_metadata({"category": "ai"})
    for result in filtered_results:
        print(f"文档: {result['document']['page_content']}")
        print(f"元数据: {result['document']['metadata']}")
        print()

    # 统计信息
    print("=== 统计信息 ===")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 保存和加载测试
    print("\n=== 保存和加载测试 ===")
    vector_store.save_to_file("test_vector_store.pkl")
    print("向量存储已保存")

    # 创建新实例并加载
    new_vector_store = create_vector_store("memory")
    new_vector_store.load_from_file("test_vector_store.pkl")
    print("向量存储已加载")

    # 验证加载的数据
    loaded_stats = new_vector_store.get_stats()
    print(f"加载后的文档数量: {loaded_stats['total_documents']}")

    print("\n测试完成！")
