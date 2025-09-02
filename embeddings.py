#!/usr/bin/env python3
"""
轻量级Embeddings类 - 可选的语义搜索功能
使用方法:
    # 基础使用
    embedder = SimpleEmbeddings()
    embeddings = embedder.embed_text("文本内容")
    
    # 相似度搜索
    results = embedder.similarity_search("查询文本", documents)
"""

import re
import math
from typing import List, Dict, Any, Union
from collections import Counter


class SimpleEmbeddings:
    """
    轻量级词袋模型嵌入 - 无需外部依赖的简单语义表示
    基于TF-IDF和词频统计，适合小型项目和学习使用
    """
    
    def __init__(self, min_word_length: int = 2, max_features: int = 10000):
        self.min_word_length = min_word_length
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf_values: Dict[str, float] = {}
        self.fitted: bool = False
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词和预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留中文、英文和数字
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        
        # 分词策略
        words = []
        
        # 英文单词
        english_words = re.findall(r'\b\w+\b', text)
        words.extend(english_words)
        
        # 中文字符（每个字符作为一个词）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        words.extend(chinese_chars)
        
        # 过滤短词（只对英文单词）
        words = [word for word in words if len(word) >= self.min_word_length or word.isalpha()]
        
        return words
    
    def fit(self, documents: List[str]):
        """构建词汇表和IDF值"""
        # 收集所有文档的词频
        doc_freq: Counter[str] = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            words = set(self._tokenize(doc))
            doc_freq.update(words)
        
        # 构建词汇表，取最常见的词
        most_common = doc_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 计算IDF值
        for word, freq in most_common:
            self.idf_values[word] = math.log(total_docs / (freq + 1))
        
        self.fitted = True
    
    def embed_text(self, text: str) -> List[float]:
        """将文本转换为向量"""
        if not self.fitted:
            raise ValueError("必须先调用fit()方法")
        
        words = self._tokenize(text)
        word_counts = Counter(words)
        
        # 创建TF-IDF向量
        vector = [0.0] * len(self.vocabulary)
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / len(words)
                idf = self.idf_values[word]
                vector[self.vocabulary[word]] = tf * idf
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            raise ValueError("向量长度不一致")
        
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(y * y for y in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def similarity_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if not self.fitted:
            self.fit(documents)
        
        query_vec = self.embed_text(query)
        doc_vecs = [self.embed_text(doc) for doc in documents]
        
        # 计算相似度
        similarities: List[Dict[str, Any]] = []
        for i, doc_vec in enumerate(doc_vecs):
            sim = self.similarity(query_vec, doc_vec)
            similarities.append({
                'index': i,
                'document': documents[i],
                'similarity': sim
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: float(x['similarity']), reverse=True)
        
        return similarities[:top_k]


class HybridSearch:
    """
    混合搜索策略 - 结合关键词匹配和语义相似度
    """
    
    def __init__(self, keyword_weight: float = 0.4, semantic_weight: float = 0.6):
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.embedder = SimpleEmbeddings()
    
    def keyword_score(self, query: str, document: str) -> float:
        """关键词匹配分数"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(query_words.intersection(doc_words))
        union = len(query_words.union(doc_words))
        
        return intersection / union if union > 0 else 0.0
    
    def search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """混合搜索"""
        if not documents:
            return []
        
        # 训练嵌入模型
        self.embedder.fit(documents)
        
        results: List[Dict[str, Union[int, str, float]]] = []
        for i, doc in enumerate(documents):
            # 计算关键词分数
            keyword_score = self.keyword_score(query, doc)
            
            # 计算语义分数
            semantic_score = 0.0
            try:
                query_vec = self.embedder.embed_text(query)
                doc_vec = self.embedder.embed_text(doc)
                semantic_score = self.embedder.similarity(query_vec, doc_vec)
            except Exception:
                pass
            
            # 加权组合分数
            combined_score = (
                self.keyword_weight * keyword_score +
                self.semantic_weight * semantic_score
            )
            
            results.append({
                'index': i,
                'document': doc,
                'keyword_score': keyword_score,
                'semantic_score': semantic_score,
                'combined_score': combined_score
            })
        
        # 按综合分数排序
        results.sort(key=lambda x: float(x['combined_score']), reverse=True)
        
        return results[:top_k]


def simple_text_search(query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    简单文本搜索 - 基于关键词匹配
    不需要训练，适合快速搜索
    """
    query_lower = query.lower()
    results: List[Dict[str, Union[int, str, float]]] = []
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        
        # 计算匹配分数
        score = 0.0
        query_words = query_lower.split()
        
        for word in query_words:
            if word in doc_lower:
                score += doc_lower.count(word)
        
        if score > 0:
            results.append({
                'index': i,
                'document': doc,
                'score': score
            })
    
            # 按分数排序
        results.sort(key=lambda x: float(x['score']), reverse=True)
    
    return results[:top_k]


# 使用示例和测试函数
if __name__ == "__main__":
    # 测试数据
    test_docs = [
        "Python是一种广泛使用的编程语言",
        "机器学习是人工智能的重要分支",
        "深度学习使用神经网络进行模式识别",
        "自然语言处理处理文本和语音数据",
        "计算机视觉让机器理解图像内容",
        "数据科学包括统计分析、数据挖掘等技术",
        "算法是解决问题的步骤和方法"
    ]
    
    print("=== 简单文本搜索 ===")
    simple_results = simple_text_search("Python 编程", test_docs)
    for result in simple_results:
        print(f"分数: {result['score']:.2f} - {result['document']}")
    
    print("\n=== 语义搜索 ===")
    embedder = SimpleEmbeddings()
    semantic_results = embedder.similarity_search("人工智能技术", test_docs, top_k=3)
    for result in semantic_results:
        print(f"相似度: {result['similarity']:.3f} - {result['document']}")
    
    print("\n=== 混合搜索 ===")
    hybrid_search = HybridSearch()
    hybrid_results = hybrid_search.search("数据 算法", test_docs, top_k=3)
    for result in hybrid_results:
        print(f"综合分数: {result['combined_score']:.3f} "
              f"(关键词: {result['keyword_score']:.3f}, "
              f"语义: {result['semantic_score']:.3f}) - "
              f"{result['document']}")