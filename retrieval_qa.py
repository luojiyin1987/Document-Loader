#!/usr/bin/env python3
"""
RetrievalQA System - 基于向量存储的问答系统
支持文档检索、上下文构建和答案生成
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 项目模块导入
from embeddings import HybridSearch, SimpleEmbeddings
from vector_store import SimpleVectorStore, create_vector_store


@dataclass
class RetrievedDocument:
    """检索到的文档对象"""

    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str
    source: str


@dataclass
class RetrievalResult:
    """检索结果"""

    query: str
    documents: List[RetrievedDocument]
    total_retrieved: int
    retrieval_time: float
    strategy: str


@dataclass
class Context:
    """构建的上下文"""

    query: str
    relevant_docs: List[RetrievedDocument]
    context_text: str
    total_tokens: int
    context_score: float


@dataclass
class AnswerResult:
    """问答结果"""

    query: str
    answer: str
    context: Context
    confidence: float
    sources: List[str]
    answer_time: float
    metadata: Dict[str, Any]


class BaseRetriever(ABC):
    """检索器抽象基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """检索相关文档"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档到检索器"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        pass


class VectorStoreRetriever(BaseRetriever):
    """基于向量存储的检索器"""

    def __init__(
        self,
        vector_store: Optional[SimpleVectorStore] = None,
        embedder: Optional[SimpleEmbeddings] = None,
    ):
        self.vector_store = vector_store or create_vector_store("memory")
        self.embedder = embedder or SimpleEmbeddings()
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """基于向量相似度检索文档"""
        import time

        start_time = time.time()

        # 生成查询嵌入
        query_embedding = self.embedder.embed_text(query)

        # 检索相似文档
        search_results = self.vector_store.search(query_embedding, top_k)

        # 转换为RetrievedDocument对象
        retrieved_docs = []
        for result in search_results:
            doc = RetrievedDocument(
                content=result["document"]["page_content"],
                metadata=result["document"]["metadata"],
                score=result["similarity"],
                doc_id=result["id"],
                source=result["document"]["metadata"].get("source", "unknown"),
            )
            retrieved_docs.append(doc)

        # 更新统计信息
        retrieval_time = time.time() - start_time
        self.retrieval_count += 1
        self.total_retrieval_time += retrieval_time

        return RetrievalResult(
            query=query,
            documents=retrieved_docs,
            total_retrieved=len(retrieved_docs),
            retrieval_time=retrieval_time,
            strategy="vector_similarity",
        )

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档到向量存储"""
        self.vector_store.add_texts(
            texts=[doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            embeddings=embeddings,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        avg_retrieval_time = self.total_retrieval_time / self.retrieval_count if self.retrieval_count > 0 else 0.0

        return {
            "retrieval_count": self.retrieval_count,
            "avg_retrieval_time": avg_retrieval_time,
            "total_retrieval_time": self.total_retrieval_time,
            "vector_store_stats": self.vector_store.get_stats(),
        }


class HybridRetriever(BaseRetriever):
    """混合检索器 - 结合向量搜索和关键词搜索"""

    def __init__(
        self,
        vector_store: Optional[SimpleVectorStore] = None,
        embedder: Optional[SimpleEmbeddings] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.vector_retriever = VectorStoreRetriever(vector_store, embedder)
        self.hybrid_search = HybridSearch()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """混合检索"""
        import time

        start_time = time.time()

        # 获取向量搜索结果
        vector_results = self.vector_retriever.retrieve(query, top_k * 2)

        # 获取所有文档内容用于关键词搜索
        all_docs = [doc.content for doc in vector_results.documents]

        # 执行关键词搜索
        keyword_results = self.hybrid_search.search(query, all_docs, top_k * 2)

        # 合并结果
        merged_docs = self._merge_results(vector_results, keyword_results, top_k)

        retrieval_time = time.time() - start_time
        self.retrieval_count += 1
        self.total_retrieval_time += retrieval_time

        return RetrievalResult(
            query=query,
            documents=merged_docs,
            total_retrieved=len(merged_docs),
            retrieval_time=retrieval_time,
            strategy="hybrid",
        )

    def _merge_results(self, vector_results: RetrievalResult, keyword_results: List[Dict], top_k: int) -> List[RetrievedDocument]:
        """合并向量搜索和关键词搜索结果"""

        # 创建文档分数映射
        doc_scores = {}

        # 添加向量搜索分数
        for doc in vector_results.documents:
            doc_scores[doc.doc_id] = {
                "doc": doc,
                "vector_score": doc.score,
                "keyword_score": 0.0,
                "combined_score": 0.0,
            }

        # 添加关键词搜索分数
        for result in keyword_results:
            # 找到对应的文档
            content = result["document"]
            for score_info in doc_scores.values():
                if score_info["doc"].page_content == content:
                    score_info["keyword_score"] = result["combined_score"]
                    break

        # 计算综合分数
        for score_info in doc_scores.values():
            vector_score = score_info.get("vector_score", 0.0)
            keyword_score = score_info.get("keyword_score", 0.0)
            combined_score = self.vector_weight * vector_score + self.keyword_weight * keyword_score
            score_info["combined_score"] = combined_score
            # 为文档对象添加分数属性
            if hasattr(score_info["doc"], "score"):
                score_info["doc"].score = combined_score
            else:
                # 如果文档对象没有score属性，我们可以将其添加到metadata中
                if not hasattr(score_info["doc"], "metadata"):
                    score_info["doc"].metadata = {}
                score_info["doc"].metadata["score"] = combined_score

        # 按综合分数排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: float(x["combined_score"]), reverse=True)

        return [item["doc"] for item in sorted_docs[:top_k]]

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档"""
        self.vector_retriever.add_documents(documents, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "retrieval_count": self.retrieval_count,
            "avg_retrieval_time": (self.total_retrieval_time / self.retrieval_count if self.retrieval_count > 0 else 0.0),
            "total_retrieval_time": self.total_retrieval_time,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "vector_retriever_stats": self.vector_retriever.get_stats(),
        }


class ContextBuilder:
    """上下文构建器"""

    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def build_context(self, query: str, retrieved_docs: List[RetrievedDocument]) -> Context:
        """构建上下文"""

        # 按分数排序文档
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.score, reverse=True)

        # 选择相关文档
        relevant_docs = self._select_relevant_docs(sorted_docs)

        # 构建上下文文本
        context_text = self._build_context_text(query, relevant_docs)

        # 估算token数量
        total_tokens = self._estimate_tokens(context_text)

        # 计算上下文分数
        context_score = self._calculate_context_score(relevant_docs)

        return Context(
            query=query,
            relevant_docs=relevant_docs,
            context_text=context_text,
            total_tokens=total_tokens,
            context_score=context_score,
        )

    def _select_relevant_docs(self, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """选择相关文档"""
        selected_docs = []
        current_tokens = 0

        for doc in docs:
            doc_tokens = self._estimate_tokens(doc.content)

            # 如果添加这个文档不会超过token限制，则添加
            if current_tokens + doc_tokens <= self.max_tokens:
                selected_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # 如果还有空间，尝试添加部分内容
                if current_tokens < self.max_tokens:
                    remaining_tokens = self.max_tokens - current_tokens
                    partial_content = self._truncate_text(doc.content, remaining_tokens)
                    if partial_content:
                        # 创建截断的文档副本
                        truncated_doc = RetrievedDocument(
                            content=partial_content,
                            metadata=doc.metadata,
                            score=doc.score * 0.8,  # 降低截断文档的分数
                            doc_id=doc.doc_id,
                            source=doc.source,
                        )
                        selected_docs.append(truncated_doc)
                        break
                break

        return selected_docs

    def _build_context_text(self, query: str, docs: List[RetrievedDocument]) -> str:
        """构建上下文文本"""
        context_parts = [f"查询: {query}\n"]
        context_parts.append("相关文档:\n")

        for i, doc in enumerate(docs, 1):
            context_parts.append(f"文档 {i} (来源: {doc.source}, 相关度: {doc.score:.3f}):")
            context_parts.append(doc.content)
            context_parts.append("\n")

        return "".join(context_parts)

    def _estimate_tokens(self, text: str) -> int:
        """估算token数量（粗略估算：1个token ≈ 4个字符）"""
        return len(text) // 4

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数量"""
        max_chars = max_tokens * 4
        return text[:max_chars]

    def _calculate_context_score(self, docs: List[RetrievedDocument]) -> float:
        """计算上下文分数"""
        if not docs:
            return 0.0

        # 基于文档分数的加权平均
        total_score = sum(doc.score for doc in docs)
        return total_score / len(docs)


class BaseAnswerGenerator(ABC):
    """答案生成器抽象基类"""

    @abstractmethod
    def generate_answer(self, query: str, context: Context) -> AnswerResult:
        """生成答案"""
        pass


class TemplateAnswerGenerator(BaseAnswerGenerator):
    """基于模板的答案生成器"""

    def __init__(self):
        self.templates = {
            "factual": "根据提供的文档信息，{answer}",
            "summary": "基于相关文档，总结如下：{answer}",
            "extraction": "从文档中提取的信息：{answer}",
            "default": "基于检索到的文档，{answer}",
        }

    def generate_answer(self, query: str, context: Context) -> AnswerResult:
        """基于模板生成答案"""
        import time

        start_time = time.time()

        # 分析查询类型
        query_type = self._analyze_query_type(query)

        # 提取关键信息
        answer_text = self._extract_answer_from_context(query, context)

        # 选择模板
        template = self.templates.get(query_type, self.templates["default"])

        # 生成最终答案
        final_answer = template.format(answer=answer_text)

        # 计算置信度
        confidence = self._calculate_confidence(context, answer_text)

        answer_time = time.time() - start_time

        return AnswerResult(
            query=query,
            answer=final_answer,
            context=context,
            confidence=confidence,
            sources=[doc.source for doc in context.relevant_docs],
            answer_time=answer_time,
            metadata={
                "query_type": query_type,
                "generator_type": "template",
                "template_used": template,
            },
        )

    def _analyze_query_type(self, query: str) -> str:
        """分析查询类型"""
        query_lower = query.lower()

        # 事实性查询
        if any(word in query_lower for word in ["是什么", "什么是", "谁", "哪里", "什么时候", "多少"]):
            return "factual"

        # 总结性查询
        if any(word in query_lower for word in ["总结", "概括", "概述", "主要内容"]):
            return "summary"

        # 提取性查询
        if any(word in query_lower for word in ["提取", "找出", "找到", "列举"]):
            return "extraction"

        return "default"

    def _extract_answer_from_context(self, query: str, context: Context) -> str:
        """从上下文中提取答案"""
        # 简单的关键词匹配答案提取
        query_words = set(query.lower().split())

        best_answer = ""
        best_score = 0

        for doc in context.relevant_docs:
            doc_text = doc.content.lower()
            score = 0

            # 计算查询词在文档中的出现频率
            for word in query_words:
                if len(word) > 1:  # 忽略单字符词
                    score += doc_text.count(word)

            # 如果找到更好的答案，更新
            if score > best_score:
                best_score = score
                best_answer = doc.content

        # 如果没有找到合适的答案，使用上下文总结
        if not best_answer:
            best_answer = f"基于检索到的 {len(context.relevant_docs)} 个文档，未找到直接回答。"

        # 限制答案长度
        if len(best_answer) > 500:
            best_answer = best_answer[:500] + "..."

        return best_answer

    def _calculate_confidence(self, context: Context, answer: str) -> float:
        """计算答案置信度"""
        # 基于上下文分数和答案长度的简单置信度计算
        context_confidence = context.context_score
        length_confidence = min(len(answer) / 200, 1.0)  # 答案长度置信度

        return context_confidence * 0.7 + length_confidence * 0.3


class RetrievalQA:
    """RetrievalQA主类"""

    def __init__(
        self,
        retriever: BaseRetriever,
        context_builder: Optional[ContextBuilder] = None,
        answer_generator: Optional[BaseAnswerGenerator] = None,
    ):
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.answer_generator = answer_generator or TemplateAnswerGenerator()

        # 统计信息
        self.total_queries = 0
        self.total_answer_time = 0.0

    def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """执行问答查询"""
        import time

        start_time = time.time()

        # 1. 检索相关文档
        retrieval_result = self.retriever.retrieve(question, top_k)

        # 2. 构建上下文
        context = self.context_builder.build_context(question, retrieval_result.documents)

        # 3. 生成答案
        answer_result = self.answer_generator.generate_answer(question, context)

        # 4. 更新统计信息
        total_time = time.time() - start_time
        self.total_queries += 1
        self.total_answer_time += total_time

        # 5. 添加总时间信息
        answer_result.metadata["total_time"] = total_time
        answer_result.metadata["retrieval_time"] = retrieval_result.retrieval_time

        return answer_result

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """添加文档到知识库"""
        if embeddings is None:
            # 如果没有提供嵌入向量，使用默认的embedder生成
            if isinstance(self.retriever, VectorStoreRetriever):
                embedder = self.retriever.embedder
            elif isinstance(self.retriever, HybridRetriever):
                embedder = self.retriever.vector_retriever.embedder
            else:
                raise ValueError("无法生成嵌入向量，请提供embeddings参数")

            # 训练embedder（如果需要）
            texts = [doc["page_content"] for doc in documents]
            embedder.fit(texts)

            # 生成嵌入向量
            embeddings = [embedder.embed_text(text) for text in texts]

        self.retriever.add_documents(documents, embeddings)

    def batch_query(self, questions: List[str], top_k: int = 5) -> List[AnswerResult]:
        """批量查询"""
        results = []
        for question in questions:
            result = self.query(question, top_k)
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        avg_answer_time = self.total_answer_time / self.total_queries if self.total_queries > 0 else 0.0

        return {
            "total_queries": self.total_queries,
            "avg_answer_time": avg_answer_time,
            "total_answer_time": self.total_answer_time,
            "retriever_stats": self.retriever.get_stats(),
            "context_builder": {
                "max_tokens": self.context_builder.max_tokens,
                "overlap_tokens": self.context_builder.overlap_tokens,
            },
            "answer_generator": {"type": type(self.answer_generator).__name__},
        }

    def save_config(self, file_path: str) -> None:
        """保存配置"""
        config = {
            "retriever_type": type(self.retriever).__name__,
            "context_builder_config": {
                "max_tokens": self.context_builder.max_tokens,
                "overlap_tokens": self.context_builder.overlap_tokens,
            },
            "answer_generator_type": type(self.answer_generator).__name__,
            "stats": self.get_stats(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def create_default(cls) -> "RetrievalQA":
        """创建默认配置的RetrievalQA实例"""
        retriever = HybridRetriever()
        context_builder = ContextBuilder(max_tokens=2000)
        answer_generator = TemplateAnswerGenerator()

        return cls(retriever, context_builder, answer_generator)


# 工厂函数
def create_retrieval_qa(
    retriever_type: str = "hybrid",
    context_max_tokens: int = 2000,
    answer_generator_type: str = "template",
) -> RetrievalQA:
    """创建RetrievalQA系统的工厂函数"""

    # 创建检索器
    if retriever_type == "vector":
        retriever = VectorStoreRetriever()
    elif retriever_type == "hybrid":
        retriever = HybridRetriever()
    else:
        raise ValueError(f"不支持的检索器类型: {retriever_type}")

    # 创建上下文构建器
    context_builder = ContextBuilder(max_tokens=context_max_tokens)

    # 创建答案生成器
    if answer_generator_type == "template":
        answer_generator = TemplateAnswerGenerator()
    else:
        raise ValueError(f"不支持的答案生成器类型: {answer_generator_type}")

    return RetrievalQA(retriever, context_builder, answer_generator)


# 使用示例和测试函数
def test_retrieval_qa():
    """测试RetrievalQA系统"""
    print("=== RetrievalQA 系统测试 ===")

    # 创建RetrievalQA实例
    qa_system = create_retrieval_qa()

    # 准备测试文档
    test_documents = [
        {
            "page_content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。" "Python以其简洁的语法和强大的功能而闻名，广泛应用于Web开发、数据分析、人工智能等领域。",
            "metadata": {
                "source": "python_doc",
                "category": "programming",
                "difficulty": "easy",
            },
        },
        {
            "page_content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。机器学习算法包括监督学习、无监督学习和强化学习等类型。",
            "metadata": {"source": "ml_doc", "category": "ai", "difficulty": "medium"},
        },
        {
            "page_content": "深度学习是机器学习的一个子领域，它使用人工神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了重大突破。",
            "metadata": {"source": "dl_doc", "category": "ai", "difficulty": "hard"},
        },
        {
            "page_content": "数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识来从数据中提取洞察。数据科学包括数据清洗、数据分析和数据可视化等步骤。",
            "metadata": {
                "source": "ds_doc",
                "category": "data",
                "difficulty": "medium",
            },
        },
    ]

    # 添加文档到知识库
    print("正在添加文档到知识库...")
    qa_system.add_documents(test_documents)
    print(f"已添加 {len(test_documents)} 个文档")

    # 测试查询
    test_questions = [
        "Python是什么时候创建的？",
        "机器学习有哪些类型？",
        "深度学习在哪些领域取得了突破？",
        "数据科学包括哪些步骤？",
    ]

    print("\n=== 测试查询 ===")
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 50)

        try:
            result = qa_system.query(question, top_k=2)

            print(f"答案: {result.answer}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"来源: {', '.join(result.sources)}")
            print(f"耗时: {result.metadata.get('total_time', 0):.3f}秒")

        except Exception as e:
            print(f"查询出错: {e}")

    # 显示统计信息
    print("\n=== 系统统计信息 ===")
    stats = qa_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n测试完成！")


if __name__ == "__main__":
    test_retrieval_qa()
