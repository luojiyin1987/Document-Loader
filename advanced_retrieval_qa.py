#!/usr/bin/env python3
"""
Advanced RetrievalQA - 高级检索增强生成系统
支持多种检索策略、上下文优化和LLM集成
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 项目模块导入
from retrieval_qa import (
    AnswerResult,
    BaseAnswerGenerator,
    BaseRetriever,
    Context,
    ContextBuilder,
    RetrievalResult,
    RetrievedDocument,
)


@dataclass
class QueryAnalysis:
    """查询分析结果"""

    query: str
    query_type: str
    intent: str
    entities: List[str]
    keywords: List[str]
    complexity: float
    time_sensitive: bool
    requires_reasoning: bool


@dataclass
class RetrievalStrategy:
    """检索策略配置"""

    name: str
    description: str
    retriever: BaseRetriever
    weight: float = 1.0
    enabled: bool = True


@dataclass
class RerankResult:
    """重排序结果"""

    documents: List[RetrievedDocument]
    rerank_scores: List[float]
    rerank_time: float
    strategy_used: str


class QueryAnalyzer:
    """查询分析器"""

    def __init__(self):
        self.query_patterns = {
            "factual": [r"是什么", r"什么是", r"谁", r"哪里", r"什么时候", r"多少", r"几个"],
            "procedural": [r"如何", r"怎么", r"怎样", r"方法", r"步骤"],
            "reasoning": [r"为什么", r"原因", r"解释", r"分析", r"比较"],
            "comparative": [r"对比", r"比较", r"区别", r"差异"],
            "predictive": [r"预测", r"未来", r"将会", r"可能"],
            "evaluative": [r"评价", r"评估", r"优缺点", r"好坏"],
        }

        self.entity_patterns = [
            r"\b[A-Z][a-z]+\b",  # 英文专有名词
            r"[\u4e00-\u9fff]+",  # 中文人名、地名等
            r"\d{4}年",  # 年份
            r"\d+%",  # 百分比
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """分析查询"""
        query_lower = query.lower()

        # 分析查询类型
        query_type = self._classify_query_type(query_lower)

        # 提取实体
        entities = self._extract_entities(query)

        # 提取关键词
        keywords = self._extract_keywords(query)

        # 分析复杂度
        complexity = self._analyze_complexity(query)

        # 判断是否时间敏感
        time_sensitive = self._is_time_sensitive(query)

        # 判断是否需要推理
        requires_reasoning = self._requires_reasoning(query_type, query)

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            intent=self._infer_intent(query_type),
            entities=entities,
            keywords=keywords,
            complexity=complexity,
            time_sensitive=time_sensitive,
            requires_reasoning=requires_reasoning,
        )

    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return query_type
        return "general"

    def _extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        entities = []
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return list(set(entities))

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r"\b\w+\b", query.lower())
        # 过滤停用词
        stop_words = {
            "的",
            "是",
            "在",
            "有",
            "和",
            "与",
            "或",
            "但",
            "如果",
            "因为",
            "所以",
            "the",
            "is",
            "at",
            "which",
            "on",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _analyze_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        factors = 0.0

        # 长度复杂度
        if len(query) > 50:
            factors += 0.2
        if len(query) > 100:
            factors += 0.2

        # 逻辑连接词
        logical_connectors = ["并且", "或者", "但是", "然而", "虽然", "and", "or", "but", "however"]
        for connector in logical_connectors:
            if connector in query:
                factors += 0.1

        # 比较和对比
        comparative_words = ["对比", "比较", "区别", "compare", "contrast", "difference"]
        for word in comparative_words:
            if word in query:
                factors += 0.2

        return min(factors, 1.0)

    def _is_time_sensitive(self, query: str) -> bool:
        """判断是否时间敏感"""
        time_words = ["最新", "最近", "当前", "现在", "latest", "recent", "current", "now"]
        return any(word in query for word in time_words)

    def _requires_reasoning(self, query_type: str, query: str) -> bool:
        """判断是否需要推理"""
        reasoning_types = ["reasoning", "comparative", "evaluative"]
        return query_type in reasoning_types

    def _infer_intent(self, query_type: str) -> str:
        """推断查询意图"""
        intent_map = {
            "factual": "获取事实信息",
            "procedural": "了解操作步骤",
            "reasoning": "理解原因和机制",
            "comparative": "对比不同选项",
            "predictive": "预测未来趋势",
            "evaluative": "评估价值和效果",
        }
        return intent_map.get(query_type, "获取一般信息")


class MultiStrategyRetriever(BaseRetriever):
    """多策略检索器"""

    def __init__(self, strategies: List[RetrievalStrategy]):
        self.strategies = [s for s in strategies if s.enabled]
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """使用多种策略检索"""
        start_time = time.time()

        # 分析查询
        query_analysis = self.query_analyzer.analyze(query)

        # 选择合适的检索策略
        selected_strategies = self._select_strategies(query_analysis)

        # 执行检索
        all_results = []
        for strategy in selected_strategies:
            try:
                result = strategy.retriever.retrieve(query, top_k)
                all_results.append((strategy, result))
            except Exception as e:
                logging.warning(f"检索策略 {strategy.name} 执行失败: {e}")

        # 合并和重排序结果
        merged_results = self._merge_and_rerank(all_results, query_analysis, top_k)

        retrieval_time = time.time() - start_time
        self.retrieval_count += 1
        self.total_retrieval_time += retrieval_time

        return RetrievalResult(
            query=query,
            documents=merged_results,
            total_retrieved=len(merged_results),
            retrieval_time=retrieval_time,
            strategy=f"multi_strategy_({len(selected_strategies)}_strategies)",
        )

    def _select_strategies(self, query_analysis: QueryAnalysis) -> List[RetrievalStrategy]:
        """根据查询分析选择检索策略"""
        selected = []

        # 基于查询类型选择策略
        if query_analysis.query_type == "factual":
            # 事实性查询优先使用向量检索
            vector_strategies = [s for s in self.strategies if "vector" in s.name.lower()]
            selected.extend(vector_strategies[:2])

        if query_analysis.query_type == "reasoning":
            # 推理性查询使用混合检索
            hybrid_strategies = [s for s in self.strategies if "hybrid" in s.name.lower()]
            selected.extend(hybrid_strategies[:2])

        # 如果没有特定策略，使用权重最高的策略
        if not selected:
            selected = sorted(self.strategies, key=lambda x: x.weight, reverse=True)[:2]

        return selected

    def _merge_and_rerank(
        self,
        all_results: List[Tuple[RetrievalStrategy, RetrievalResult]],
        query_analysis: QueryAnalysis,
        top_k: int,
    ) -> List[RetrievedDocument]:
        """合并和重排序检索结果"""

        # 收集所有文档
        all_docs = {}
        for strategy, result in all_results:
            for doc in result.documents:
                if doc.doc_id not in all_docs:
                    all_docs[doc.doc_id] = {"doc": doc, "strategies": [], "scores": []}
                all_docs[doc.doc_id]["strategies"].append(strategy.name)
                all_docs[doc.doc_id]["scores"].append(
                    float(doc.score if hasattr(doc, "score") else 0.0)
                )

        # 重排序
        reranked_docs = []
        for doc_id, doc_info in all_docs.items():
            doc = doc_info["doc"]

            # 计算重排序分数
            rerank_score = self._calculate_rerank_score(doc, doc_info, query_analysis)

            # 创建重排序后的文档
            reranked_doc = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=rerank_score,
                doc_id=doc.doc_id,
                source=doc.source,
            )

            # 添加策略信息到元数据
            reranked_doc.metadata["retrieval_strategies"] = doc_info["strategies"]
            reranked_doc.metadata["original_scores"] = doc_info["scores"]

            reranked_docs.append(reranked_doc)

        # 按重排序分数排序
        reranked_docs.sort(key=lambda x: x.score, reverse=True)

        return reranked_docs[:top_k]

    def _calculate_rerank_score(
        self, doc: RetrievedDocument, doc_info: Dict, query_analysis: QueryAnalysis
    ) -> float:
        """计算重排序分数"""

        # 基础分数（平均分数）
        base_score = sum(doc_info["scores"]) / len(doc_info["scores"])

        # 策略多样性奖励
        diversity_bonus = len(set(doc_info["strategies"])) * 0.1

        # 查询-文档相关性
        relevance_score = self._calculate_relevance(doc, query_analysis)

        # 最终分数
        final_score = base_score * 0.6 + relevance_score * 0.3 + diversity_bonus * 0.1

        return final_score

    def _calculate_relevance(self, doc: RetrievedDocument, query_analysis: QueryAnalysis) -> float:
        """计算查询-文档相关性"""
        score = 0.0

        # 实体匹配
        doc_content = doc.content.lower()
        for entity in query_analysis.entities:
            if entity.lower() in doc_content:
                score += 0.2

        # 关键词匹配
        for keyword in query_analysis.keywords:
            if keyword in doc_content:
                score += 0.1

        return min(score, 1.0)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档到所有检索器"""
        for strategy in self.strategies:
            strategy.retriever.add_documents(documents, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "retrieval_count": self.retrieval_count,
            "avg_retrieval_time": (
                self.total_retrieval_time / self.retrieval_count
                if self.retrieval_count > 0
                else 0.0
            ),
            "total_retrieval_time": self.total_retrieval_time,
            "strategies": [
                {
                    "name": strategy.name,
                    "weight": strategy.weight,
                    "enabled": strategy.enabled,
                    "stats": strategy.retriever.get_stats(),
                }
                for strategy in self.strategies
            ],
        }


class AdvancedContextBuilder(ContextBuilder):
    """高级上下文构建器"""

    def __init__(self, max_tokens: int = 3000, overlap_tokens: int = 200):
        super().__init__(max_tokens, overlap_tokens)
        self.context_strategies = {
            "relevant_first": self._relevant_first_strategy,
            "diverse": self._diverse_strategy,
            "comprehensive": self._comprehensive_strategy,
        }
        self.current_strategy = "relevant_first"

    def build_context(self, query: str, retrieved_docs: List[RetrievedDocument]) -> Context:
        """构建上下文"""
        # 选择上下文构建策略
        strategy_func = self.context_strategies[self.current_strategy]
        selected_docs = strategy_func(query, retrieved_docs)

        # 构建上下文文本
        context_text = self._build_advanced_context_text(query, selected_docs)

        # 估算token数量
        total_tokens = self._estimate_tokens(context_text)

        # 计算上下文分数
        context_score = self._calculate_advanced_context_score(selected_docs)

        return Context(
            query=query,
            relevant_docs=selected_docs,
            context_text=context_text,
            total_tokens=total_tokens,
            context_score=context_score,
        )

    def _relevant_first_strategy(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """相关优先策略"""
        return self._select_relevant_docs(docs)

    def _diverse_strategy(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """多样性策略"""
        selected_docs = []
        used_sources = set()

        # 按分数排序
        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

        for doc in sorted_docs:
            if doc.source not in used_sources:
                selected_docs.append(doc)
                used_sources.add(doc.source)

                # 检查token限制
                if (
                    self._estimate_tokens(self._build_context_text(query, selected_docs))
                    > self.max_tokens
                ):
                    selected_docs.pop()
                    break

        return selected_docs

    def _comprehensive_strategy(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """全面策略"""
        # 选择高相关性的文档，同时确保覆盖不同来源
        high_relevance_docs = [doc for doc in docs if doc.score > 0.7]
        medium_relevance_docs = [doc for doc in docs if 0.5 <= doc.score <= 0.7]

        selected_docs = []

        # 优先选择高相关性文档
        selected_docs.extend(high_relevance_docs)

        # 补充中等相关性文档以增加多样性
        for doc in medium_relevance_docs:
            if doc.source not in [d.source for d in selected_docs]:
                selected_docs.append(doc)

                if (
                    self._estimate_tokens(self._build_context_text(query, selected_docs))
                    > self.max_tokens
                ):
                    selected_docs.pop()
                    break

        return selected_docs

    def _build_advanced_context_text(self, query: str, docs: List[RetrievedDocument]) -> str:
        """构建高级上下文文本"""
        context_parts = [f"查询: {query}\n"]
        context_parts.append("相关文档:\n")

        for i, doc in enumerate(docs, 1):
            context_parts.append(f"文档 {i}:")
            context_parts.append(f"来源: {doc.source}")
            context_parts.append(f"相关度: {doc.score:.3f}")

            # 添加元数据信息
            if doc.metadata:
                metadata_str = ", ".join(
                    [
                        f"{k}: {v}"
                        for k, v in doc.metadata.items()
                        if k not in ["source", "retrieval_strategies"]
                    ]
                )
                if metadata_str:
                    context_parts.append(f"元数据: {metadata_str}")

            context_parts.append("内容:")
            context_parts.append(doc.content)
            context_parts.append("\n")

        return "".join(context_parts)

    def _calculate_advanced_context_score(self, docs: List[RetrievedDocument]) -> float:
        """计算高级上下文分数"""
        if not docs:
            return 0.0

        # 基于多个因素的加权分数
        scores = []

        for doc in docs:
            doc_score = doc.score

            # 长度奖励
            length_bonus = min(len(doc.content) / 500, 1.0) * 0.1

            # 来源多样性奖励
            diversity_bonus = 0.0
            if hasattr(doc.metadata, "get") and doc.metadata.get("retrieval_strategies"):
                diversity_bonus = len(set(doc.metadata["retrieval_strategies"])) * 0.05

            scores.append(doc_score + length_bonus + diversity_bonus)

        return sum(scores) / len(scores)

    def set_strategy(self, strategy: str):
        """设置上下文构建策略"""
        if strategy in self.context_strategies:
            self.current_strategy = strategy
        else:
            raise ValueError(f"不支持的策略: {strategy}")


class LLMAnswerGenerator(BaseAnswerGenerator):
    """基于LLM的答案生成器"""

    def __init__(self, model_name: str = "default", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.prompts = {
            "qa": """基于以下文档信息回答问题。请确保答案准确、相关且基于提供的文档。

查询: {query}

相关文档:
{context}

请回答:""",
            "reasoning": """基于以下文档信息回答问题，并提供详细的推理过程。

查询: {query}

相关文档:
{context}

请提供详细的答案和推理过程:""",
            "summary": """基于以下文档信息，总结回答问题。

查询: {query}

相关文档:
{context}

请提供总结性的答案:""",
        }

    def generate_answer(self, query: str, context: Context) -> AnswerResult:
        """生成答案"""
        start_time = time.time()

        # 选择合适的提示模板
        prompt_template = self._select_prompt_template(query)

        # 构建提示
        prompt = prompt_template.format(query=query, context=context.context_text)

        # 生成答案（这里使用模拟的LLM调用）
        answer_text = self._call_llm(prompt)

        # 计算置信度
        confidence = self._calculate_llm_confidence(context, answer_text)

        answer_time = time.time() - start_time

        return AnswerResult(
            query=query,
            answer=answer_text,
            context=context,
            confidence=confidence,
            sources=[doc.source for doc in context.relevant_docs],
            answer_time=answer_time,
            metadata={
                "model": self.model_name,
                "prompt_template": prompt_template,
                "context_tokens": context.total_tokens,
                "answer_tokens": self._estimate_tokens(answer_text),
            },
        )

    def _select_prompt_template(self, query: str) -> str:
        """选择提示模板"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["为什么", "原因", "解释", "分析"]):
            return self.prompts["reasoning"]
        elif any(word in query_lower for word in ["总结", "概括", "概述"]):
            return self.prompts["summary"]
        else:
            return self.prompts["qa"]

    def _call_llm(self, prompt: str) -> str:
        """调用LLM（模拟实现）"""
        # 这里是一个模拟的LLM调用
        # 实际使用时可以替换为真实的LLM API调用

        # 模拟处理延迟
        time.sleep(0.1)

        # 基于提示生成简单的回答
        if "总结" in prompt or "概括" in prompt:
            return "基于相关文档的内容，这是一个关于文档主题的总结性回答。"
        elif "为什么" in prompt or "原因" in prompt:
            return "根据文档内容，原因可以归纳为以下几点：1) 主要原因；2) 次要原因；3) 其他因素。"
        else:
            return "根据提供的文档信息，这是一个基于事实的回答。文档中包含了相关的详细信息。"

    def _calculate_llm_confidence(self, context: Context, answer: str) -> float:
        """计算LLM答案置信度"""
        # 基于上下文分数和答案质量的综合置信度
        context_confidence = context.context_score

        # 答案长度相关性
        length_score = min(len(answer) / 200, 1.0) * 0.2

        # 上下文利用率
        context_usage = min(context.total_tokens / 2000, 1.0) * 0.2

        return min(context_confidence * 0.6 + length_score + context_usage, 1.0)

    def _estimate_tokens(self, text: str) -> int:
        """估算token数量"""
        return len(text) // 4


class AdvancedRetrievalQA:
    """高级RetrievalQA系统"""

    def __init__(
        self,
        retriever: MultiStrategyRetriever,
        context_builder: AdvancedContextBuilder,
        answer_generator: BaseAnswerGenerator,
        enable_reranking: bool = True,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.enable_reranking = enable_reranking

        # 性能监控
        self.performance_metrics = {
            "total_queries": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "cache_hits": 0,
            "error_count": 0,
        }

        # 查询缓存
        self.query_cache: Dict[str, AnswerResult] = {}

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_cache: bool = True,
        context_strategy: Optional[str] = None,
    ) -> AnswerResult:
        """执行高级问答查询"""
        start_time = time.time()

        # 检查缓存
        if use_cache and question in self.query_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.query_cache[question]

        try:
            # 设置上下文策略
            if context_strategy:
                self.context_builder.set_strategy(context_strategy)

            # 1. 检索
            retrieval_result = self.retriever.retrieve(question, top_k)

            # 2. 构建上下文
            context = self.context_builder.build_context(question, retrieval_result.documents)

            # 3. 生成答案
            answer_result = self.answer_generator.generate_answer(question, context)

            # 4. 添加性能信息
            total_time = time.time() - start_time
            answer_result.metadata["total_time"] = total_time
            answer_result.metadata["retrieval_time"] = retrieval_result.retrieval_time

            # 5. 更新性能指标
            self._update_performance_metrics(total_time, False)

            # 6. 缓存结果
            if use_cache:
                self.query_cache[question] = answer_result

            return answer_result

        except Exception as e:
            self._update_performance_metrics(time.time() - start_time, True)
            logging.error(f"查询处理失败: {e}")
            raise

    def batch_query(self, questions: List[str], top_k: int = 5) -> List[AnswerResult]:
        """批量查询"""
        results = []
        for question in questions:
            try:
                result = self.query(question, top_k)
                results.append(result)
            except Exception as e:
                logging.error(f"批量查询失败: {question}, 错误: {e}")
                # 添加错误结果
                results.append(self._create_error_result(question, str(e)))

        return results

    def _create_error_result(self, question: str, error_msg: str) -> AnswerResult:
        """创建错误结果"""
        return AnswerResult(
            query=question,
            answer=f"查询处理失败: {error_msg}",
            context=Context(
                query=question, relevant_docs=[], context_text="", total_tokens=0, context_score=0.0
            ),
            confidence=0.0,
            sources=[],
            answer_time=0.0,
            metadata={"error": error_msg},
        )

    def _update_performance_metrics(self, time_taken: float, is_error: bool):
        """更新性能指标"""
        self.performance_metrics["total_queries"] += 1
        self.performance_metrics["total_time"] += time_taken
        self.performance_metrics["avg_time"] = (
            self.performance_metrics["total_time"] / self.performance_metrics["total_queries"]
        )

        if is_error:
            self.performance_metrics["error_count"] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            "cache_size": len(self.query_cache),
            "error_rate": (
                self.performance_metrics["error_count"] / self.performance_metrics["total_queries"]
                if self.performance_metrics["total_queries"] > 0
                else 0.0
            ),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / self.performance_metrics["total_queries"]
                if self.performance_metrics["total_queries"] > 0
                else 0.0
            ),
        }

    def clear_cache(self):
        """清空缓存"""
        self.query_cache.clear()

    def add_documents(
        self, documents: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """添加文档"""
        if embeddings is None:
            # 如果没有提供嵌入向量，需要生成
            embeddings = self._generate_embeddings(documents)

        self.retriever.add_documents(documents, embeddings)
        # 清空缓存，因为知识库已更新
        self.clear_cache()

    def _generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """生成文档嵌入向量"""
        # 获取embedder
        embedder = None

        # 尝试从策略中获取embedder
        for strategy in self.retriever.strategies:
            if hasattr(strategy.retriever, "embedder"):
                embedder = strategy.retriever.embedder
                break
            elif hasattr(strategy.retriever, "vector_retriever"):
                embedder = strategy.retriever.vector_retriever.embedder
                break

        if embedder is None:
            raise ValueError("无法获取embedder来生成嵌入向量")

        # 训练embedder（如果需要）
        texts = [doc["page_content"] for doc in documents]
        if hasattr(embedder, "fit") and not getattr(embedder, "fitted", False):
            embedder.fit(texts)

        # 生成嵌入向量
        embeddings = []
        for text in texts:
            try:
                embedding = embedder.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"生成嵌入向量失败: {e}")
                # 生成零向量作为fallback
                embedding_dim = 137  # 默认维度
                embeddings.append([0.0] * embedding_dim)

        return embeddings

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "performance": self.get_performance_metrics(),
            "retriever": self.retriever.get_stats(),
            "context_builder": {
                "strategy": self.context_builder.current_strategy,
                "max_tokens": self.context_builder.max_tokens,
                "overlap_tokens": self.context_builder.overlap_tokens,
            },
            "answer_generator": {
                "type": type(self.answer_generator).__name__,
                "model": getattr(self.answer_generator, "model_name", "unknown"),
            },
        }


def create_advanced_retrieval_qa() -> AdvancedRetrievalQA:
    """创建高级RetrievalQA系统"""
    from retrieval_qa import HybridRetriever, VectorStoreRetriever

    # 创建多个检索策略
    vector_retriever = VectorStoreRetriever()
    hybrid_retriever = HybridRetriever()

    strategies = [
        RetrievalStrategy(
            name="vector_similarity",
            description="基于向量的相似度检索",
            retriever=vector_retriever,
            weight=0.6,
        ),
        RetrievalStrategy(
            name="hybrid_search",
            description="混合检索（向量+关键词）",
            retriever=hybrid_retriever,
            weight=0.8,
        ),
    ]

    # 创建多策略检索器
    multi_strategy_retriever = MultiStrategyRetriever(strategies)

    # 创建高级上下文构建器
    context_builder = AdvancedContextBuilder(max_tokens=3000)

    # 创建LLM答案生成器
    answer_generator = LLMAnswerGenerator()

    # 创建高级RetrievalQA系统
    return AdvancedRetrievalQA(
        retriever=multi_strategy_retriever,
        context_builder=context_builder,
        answer_generator=answer_generator,
        enable_reranking=True,
    )


def test_advanced_retrieval_qa():
    """测试高级RetrievalQA系统"""
    print("=== 高级RetrievalQA 系统测试 ===")

    # 创建系统
    qa_system = create_advanced_retrieval_qa()

    # 准备测试文档
    test_documents = [
        {
            "page_content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
            "AI包括机器学习、深度学习、自然语言处理等多个子领域。",
            "metadata": {"source": "ai_overview", "category": "ai", "difficulty": "medium"},
        },
        {
            "page_content": "机器学习是AI的核心技术之一，它使计算机能够从数据中学习并改进性能。"
            "主要类型包括监督学习、无监督学习和强化学习。常见的算法有决策树、神经网络、支持向量机等。",
            "metadata": {"source": "ml_guide", "category": "ai", "difficulty": "medium"},
        },
        {
            "page_content": "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。"
            "它在图像识别、语音识别、自然语言处理等领域取得了突破性进展。",
            "metadata": {"source": "dl_introduction", "category": "ai", "difficulty": "hard"},
        },
        {
            "page_content": "自然语言处理（NLP）是AI的重要分支，专注于计算机与人类语言之间的交互。NLP技术包括机器翻译、情感分析、文本摘要、问答系统等应用。",
            "metadata": {"source": "nlp_basics", "category": "nlp", "difficulty": "medium"},
        },
        {
            "page_content": "计算机视觉是AI的另一个重要领域，致力于使计算机能够理解和解释视觉信息。应用包括人脸识别、物体检测、图像分割、自动驾驶等。",
            "metadata": {"source": "cv_applications", "category": "cv", "difficulty": "hard"},
        },
    ]

    # 添加文档
    print("正在添加文档到知识库...")
    qa_system.add_documents(test_documents)
    print(f"已添加 {len(test_documents)} 个文档")

    # 测试查询
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些主要类型？",
        "深度学习在哪些领域取得了突破？",
        "自然语言处理有哪些应用？",
        "分析计算机视觉的技术特点",
        "对比机器学习和深度学习的区别",
    ]

    print("\n=== 测试查询 ===")
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 60)

        try:
            result = qa_system.query(question, top_k=3, context_strategy="relevant_first")

            print(f"答案: {result.answer}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"来源: {', '.join(set(result.sources))}")
            print(f"上下文文档数: {len(result.context.relevant_docs)}")
            print(f"总耗时: {result.metadata.get('total_time', 0):.3f}秒")

        except Exception as e:
            print(f"查询出错: {e}")

    # 显示系统统计信息
    print("\n=== 系统统计信息 ===")
    stats = qa_system.get_system_stats()
    print(f"总查询数: {stats['performance']['total_queries']}")
    print(f"平均响应时间: {stats['performance']['avg_time']:.3f}秒")
    print(f"缓存命中率: {stats['performance']['cache_hit_rate']:.2%}")
    print(f"错误率: {stats['performance']['error_rate']:.2%}")

    print("\n=== 检索器统计 ===")
    retriever_stats = stats["retriever"]
    print(f"检索次数: {retriever_stats['retrieval_count']}")
    print(f"平均检索时间: {retriever_stats['avg_retrieval_time']:.3f}秒")

    print("\n测试完成！")


if __name__ == "__main__":
    test_advanced_retrieval_qa()
