#!/usr/bin/env python3
"""
RetrievalQA 评估和优化模块
提供系统性能评估、质量度量和优化建议
"""

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 导入RetrievalQA模块
from retrieval_qa import AnswerResult


@dataclass
class EvaluationMetric:
    """评估指标"""

    name: str
    value: float
    description: str
    threshold: Optional[float] = None
    unit: str = ""


@dataclass
class QueryEvaluation:
    """单个查询评估结果"""

    query: str
    expected_answer: Optional[str]
    actual_answer: str
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    response_time: float
    context_quality: float
    source_diversity: float
    overall_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEvaluation:
    """系统评估结果"""

    total_queries: int
    avg_relevance: float
    avg_accuracy: float
    avg_completeness: float
    avg_response_time: float
    avg_context_quality: float
    avg_source_diversity: float
    overall_score: float
    metrics: List[EvaluationMetric] = field(default_factory=list)
    query_evaluations: List[QueryEvaluation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evaluation_time: datetime = field(default_factory=datetime.now)


class RelevanceEvaluator:
    """相关性评估器"""

    def __init__(self):
        self.keyword_weight = 0.4
        self.semantic_weight = 0.6

    def evaluate_relevance(self, query: str, answer: str, context: str) -> float:
        """评估答案与查询的相关性"""
        # 关键词匹配分数
        keyword_score = self._keyword_relevance(query, answer)

        # 语义相关性分数
        semantic_score = self._semantic_relevance(query, answer, context)

        # 综合分数
        return keyword_score * self.keyword_weight + semantic_score * self.semantic_weight

    def _keyword_relevance(self, query: str, answer: str) -> float:
        """计算关键词相关性"""
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        answer_words = re.findall(r"\b\w+\b", answer.lower())

        if not query_words:
            return 0.0

        # 计算查询词在答案中的覆盖率
        matched_words = sum(1 for word in query_words if word in answer_words)
        coverage = matched_words / len(query_words)

        # 计算词频权重
        frequency_score = sum(answer_words.count(word) for word in query_words) / len(answer_words)

        return min(coverage * 0.7 + frequency_score * 0.3, 1.0)

    def _semantic_relevance(self, query: str, answer: str, context: str) -> float:
        """计算语义相关性"""
        # 简化的语义相关性计算
        answer_lower = answer.lower()

        score = 0.0

        # 检查是否直接回答了问题
        if any(word in answer_lower for word in ["是", "不是", "有", "没有", "包括", "包含"]):
            score += 0.3

        # 检查答案长度是否合理
        if 50 <= len(answer) <= 500:
            score += 0.2

        # 检查是否包含解释性内容
        if any(word in answer_lower for word in ["因为", "所以", "由于", "原因", "例如"]):
            score += 0.2

        # 检查上下文利用率
        context_words = set(re.findall(r"\b\w+\b", context.lower()))
        answer_words = set(re.findall(r"\b\w+\b", answer_lower))

        if context_words and answer_words:
            overlap = len(context_words.intersection(answer_words)) / len(answer_words)
            score += min(overlap, 0.3)

        return min(score, 1.0)


class AccuracyEvaluator:
    """准确性评估器"""

    def evaluate_accuracy(self, answer: str, expected_answer: Optional[str] = None) -> float:
        """评估答案准确性"""
        if not expected_answer:
            # 如果没有期望答案，基于答案质量评估
            return self._evaluate_answer_quality(answer)

        # 有期望答案时进行对比
        return self._compare_answers(answer, expected_answer)

    def _evaluate_answer_quality(self, answer: str) -> float:
        """基于答案质量评估准确性"""
        score = 0.0

        # 检查答案是否合理
        if len(answer) > 20:  # 答案长度合理
            score += 0.2

        # 检查是否包含具体信息
        if re.search(r"\d+", answer):  # 包含数字
            score += 0.2

        # 检查结构是否清晰
        if any(marker in answer for marker in ["：", "，", "。", "；", "\n"]):
            score += 0.2

        # 检查是否避免模糊表述
        vague_phrases = ["可能", "大概", "也许", "似乎", "差不多"]
        vague_count = sum(1 for phrase in vague_phrases if phrase in answer)
        if vague_count <= 1:
            score += 0.2

        # 检查语言表达
        if not re.search(r"(.)\1{3,}", answer):  # 避免重复字符
            score += 0.2

        return min(score, 1.0)

    def _compare_answers(self, answer: str, expected_answer: str) -> float:
        """对比答案与期望答案"""
        # 简化的答案对比
        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        expected_words = set(re.findall(r"\b\w+\b", expected_answer.lower()))

        if not expected_words:
            return 0.0

        # 计算词汇重叠度
        overlap = len(answer_words.intersection(expected_words))
        precision = overlap / len(answer_words) if answer_words else 0
        recall = overlap / len(expected_words)

        # F1分数
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        return f1_score


class CompletenessEvaluator:
    """完整性评估器"""

    def evaluate_completeness(self, query: str, answer: str, context: str) -> float:
        """评估答案完整性"""
        score = 0.0

        # 基于查询类型的完整性检查
        query_type = self._classify_query_type(query)

        if query_type == "factual":
            score += self._check_factual_completeness(answer)
        elif query_type == "procedural":
            score += self._check_procedural_completeness(answer)
        elif query_type == "explanatory":
            score += self._check_explanatory_completeness(answer)
        elif query_type == "comparative":
            score += self._check_comparative_completeness(answer)
        else:
            score += self._check_general_completeness(answer)

        # 检查信息密度
        score += self._check_information_density(answer, context)

        return min(score, 1.0)

    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["是什么", "什么是", "谁", "哪里", "什么时候"]):
            return "factual"
        elif any(word in query_lower for word in ["如何", "怎么", "步骤", "方法"]):
            return "procedural"
        elif any(word in query_lower for word in ["为什么", "原因", "解释", "分析"]):
            return "explanatory"
        elif any(word in query_lower for word in ["对比", "比较", "区别"]):
            return "comparative"
        else:
            return "general"

    def _check_factual_completeness(self, answer: str) -> float:
        """检查事实性答案完整性"""
        score = 0.0

        # 检查是否包含具体信息
        if re.search(r"\d+", answer):  # 包含数字
            score += 0.3

        # 检查是否包含关键信息点
        if len(answer.split("。")) >= 2:  # 至少两个句子
            score += 0.4

        # 检查是否避免过于笼统
        vague_words = ["一些", "某些", "一些方面"]
        if not any(word in answer for word in vague_words):
            score += 0.3

        return score

    def _check_procedural_completeness(self, answer: str) -> float:
        """检查程序性答案完整性"""
        score = 0.0

        # 检查是否包含步骤指示
        step_indicators = ["首先", "然后", "接着", "最后", "第一步", "第二步"]
        if any(indicator in answer for indicator in step_indicators):
            score += 0.4

        # 检查步骤数量
        sentences = answer.split("。")
        if len(sentences) >= 3:
            score += 0.3

        # 检查逻辑连贯性
        if "然后" in answer or "接着" in answer:
            score += 0.3

        return score

    def _check_explanatory_completeness(self, answer: str) -> float:
        """检查解释性答案完整性"""
        score = 0.0

        # 检查是否包含因果关系
        causal_words = ["因为", "所以", "由于", "导致", "原因", "结果"]
        if any(word in answer for word in causal_words):
            score += 0.4

        # 检查是否包含例子
        example_words = ["例如", "比如", "如", "举例"]
        if any(word in answer for word in example_words):
            score += 0.3

        # 检查解释深度
        if len(answer) > 100:  # 足够的长度
            score += 0.3

        return score

    def _check_comparative_completeness(self, answer: str) -> float:
        """检查比较性答案完整性"""
        score = 0.0

        # 检查是否包含对比词
        comparison_words = ["对比", "比较", "区别", "差异", "相同", "不同"]
        if any(word in answer for word in comparison_words):
            score += 0.4

        # 检查是否涵盖多个方面
        aspects = ["优点", "缺点", "特点", "适用场景"]
        covered_aspects = sum(1 for aspect in aspects if aspect in answer)
        score += min(covered_aspects * 0.2, 0.6)

        return score

    def _check_general_completeness(self, answer: str) -> float:
        """检查一般答案完整性"""
        score = 0.0

        # 基于长度和结构
        if len(answer) > 50:
            score += 0.3

        # 基于句子数量
        sentences = answer.split("。")
        if len(sentences) >= 2:
            score += 0.3

        # 基于信息密度
        if len(set(answer.split())) > 20:  # 词汇多样性
            score += 0.4

        return score

    def _check_information_density(self, answer: str, context: str) -> float:
        """检查信息密度"""
        if not context:
            return 0.0

        # 计算答案相对于上下文的信息密度
        answer_unique_words = set(re.findall(r"\b\w+\b", answer.lower()))
        context_words = set(re.findall(r"\b\w+\b", context.lower()))

        if context_words:
            density = len(answer_unique_words) / len(context_words)
            return min(density * 2, 1.0)  # 2倍权重，最大1.0

        return 0.0


class ContextQualityEvaluator:
    """上下文质量评估器"""

    def evaluate_context_quality(self, context_docs: List[Any], answer: str) -> float:
        """评估上下文质量"""
        if not context_docs:
            return 0.0

        scores = []

        for doc in context_docs:
            doc_score = 0.0

            # 文档相关性
            if hasattr(doc, "score"):
                doc_score += doc.score * 0.4

            # 文档长度合理性
            if hasattr(doc, "content"):
                content_length = len(doc.content)
                if 100 <= content_length <= 1000:
                    doc_score += 0.3
                elif content_length > 1000:
                    doc_score += 0.2

            # 文档多样性
            if hasattr(doc, "metadata") and doc.metadata:
                doc_score += 0.3

            scores.append(min(doc_score, 1.0))

        return statistics.mean(scores) if scores else 0.0


class SourceDiversityEvaluator:
    """来源多样性评估器"""

    def evaluate_source_diversity(self, context_docs: List[Any]) -> float:
        """评估来源多样性"""
        if not context_docs:
            return 0.0

        # 收集来源信息
        sources = []
        for doc in context_docs:
            if hasattr(doc, "source"):
                sources.append(doc.source)
            elif hasattr(doc, "metadata") and doc.metadata:
                source = doc.metadata.get("source", "unknown")
                sources.append(source)

        if not sources:
            return 0.0

        # 计算多样性指标
        unique_sources = len(set(sources))
        total_sources = len(sources)

        # 多样性分数
        diversity_ratio = unique_sources / total_sources

        # 奖励更多样的来源
        if unique_sources >= 3:
            diversity_ratio += 0.2

        return min(diversity_ratio, 1.0)


class RetrievalQAEvaluator:
    """RetrievalQA系统评估器"""

    def __init__(self):
        self.relevance_evaluator = RelevanceEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.completeness_evaluator = CompletenessEvaluator()
        self.context_quality_evaluator = ContextQualityEvaluator()
        self.source_diversity_evaluator = SourceDiversityEvaluator()

    def evaluate_query(
        self, query: str, answer_result: AnswerResult, expected_answer: Optional[str] = None
    ) -> QueryEvaluation:
        """评估单个查询"""
        # 提取上下文信息
        context_docs = answer_result.context.relevant_docs
        context_text = answer_result.context.context_text

        # 计算各项指标
        relevance_score = self.relevance_evaluator.evaluate_relevance(
            query, answer_result.answer, context_text
        )

        accuracy_score = self.accuracy_evaluator.evaluate_accuracy(
            answer_result.answer, expected_answer
        )

        completeness_score = self.completeness_evaluator.evaluate_completeness(
            query, answer_result.answer, context_text
        )

        context_quality = self.context_quality_evaluator.evaluate_context_quality(
            context_docs, answer_result.answer
        )

        source_diversity = self.source_diversity_evaluator.evaluate_source_diversity(context_docs)

        # 计算总体分数
        overall_score = (
            relevance_score * 0.3
            + accuracy_score * 0.3
            + completeness_score * 0.2
            + context_quality * 0.1
            + source_diversity * 0.1
        )

        return QueryEvaluation(
            query=query,
            expected_answer=expected_answer,
            actual_answer=answer_result.answer,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            response_time=answer_result.answer_time,
            context_quality=context_quality,
            source_diversity=source_diversity,
            overall_score=overall_score,
            details={
                "confidence": answer_result.confidence,
                "sources": answer_result.sources,
                "context_docs_count": len(context_docs),
                "total_tokens": answer_result.context.total_tokens,
            },
        )

    def evaluate_system(self, qa_system, test_queries: List[Dict[str, Any]]) -> SystemEvaluation:
        """评估整个系统"""
        query_evaluations = []

        print(f"开始评估 {len(test_queries)} 个测试查询...")

        for i, test_case in enumerate(test_queries, 1):
            print(f"评估查询 {i}/{len(test_queries)}: {test_case['query']}")

            try:
                # 执行查询
                start_time = time.time()
                answer_result = qa_system.query(test_case["query"])
                end_time = time.time()

                # 更新响应时间
                answer_result.answer_time = end_time - start_time

                # 评估查询
                evaluation = self.evaluate_query(
                    test_case["query"], answer_result, test_case.get("expected_answer")
                )

                query_evaluations.append(evaluation)

            except Exception as e:
                print(f"查询评估失败: {test_case['query']}, 错误: {e}")
                continue

        # 计算系统级指标
        if not query_evaluations:
            raise ValueError("没有成功评估任何查询")

        avg_relevance = statistics.mean([qe.relevance_score for qe in query_evaluations])
        avg_accuracy = statistics.mean([qe.accuracy_score for qe in query_evaluations])
        avg_completeness = statistics.mean([qe.completeness_score for qe in query_evaluations])
        avg_response_time = statistics.mean([qe.response_time for qe in query_evaluations])
        avg_context_quality = statistics.mean([qe.context_quality for qe in query_evaluations])
        avg_source_diversity = statistics.mean([qe.source_diversity for qe in query_evaluations])
        overall_score = statistics.mean([qe.overall_score for qe in query_evaluations])

        # 生成指标
        metrics = self._generate_metrics(query_evaluations)

        # 生成建议
        recommendations = self._generate_recommendations(query_evaluations)

        return SystemEvaluation(
            total_queries=len(query_evaluations),
            avg_relevance=avg_relevance,
            avg_accuracy=avg_accuracy,
            avg_completeness=avg_completeness,
            avg_response_time=avg_response_time,
            avg_context_quality=avg_context_quality,
            avg_source_diversity=avg_source_diversity,
            overall_score=overall_score,
            metrics=metrics,
            query_evaluations=query_evaluations,
            recommendations=recommendations,
        )

    def _generate_metrics(self, query_evaluations: List[QueryEvaluation]) -> List[EvaluationMetric]:
        """生成详细指标"""
        metrics = []

        # 基础指标
        metrics.append(
            EvaluationMetric(
                "相关性",
                statistics.mean([qe.relevance_score for qe in query_evaluations]),
                "答案与查询的相关程度",
                0.7,
            )
        )

        metrics.append(
            EvaluationMetric(
                "准确性",
                statistics.mean([qe.accuracy_score for qe in query_evaluations]),
                "答案的准确程度",
                0.75,
            )
        )

        metrics.append(
            EvaluationMetric(
                "完整性",
                statistics.mean([qe.completeness_score for qe in query_evaluations]),
                "答案的完整程度",
                0.7,
            )
        )

        metrics.append(
            EvaluationMetric(
                "响应时间",
                statistics.mean([qe.response_time for qe in query_evaluations]),
                "平均响应时间",
                2.0,
                "秒",
            )
        )

        metrics.append(
            EvaluationMetric(
                "上下文质量",
                statistics.mean([qe.context_quality for qe in query_evaluations]),
                "检索上下文的质量",
                0.7,
            )
        )

        metrics.append(
            EvaluationMetric(
                "来源多样性",
                statistics.mean([qe.source_diversity for qe in query_evaluations]),
                "信息来源的多样性",
                0.6,
            )
        )

        # 性能指标
        response_times = [qe.response_time for qe in query_evaluations]
        metrics.append(
            EvaluationMetric(
                "响应时间稳定性",
                statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "响应时间的稳定性",
                0.5,
                "秒",
            )
        )

        # 质量分布
        high_quality_count = sum(1 for qe in query_evaluations if qe.overall_score >= 0.8)
        metrics.append(
            EvaluationMetric(
                "高质量答案比例",
                high_quality_count / len(query_evaluations),
                "高质量答案（分数≥0.8）的比例",
                0.6,
            )
        )

        return metrics

    def _generate_recommendations(self, query_evaluations: List[QueryEvaluation]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 分析各项指标
        avg_relevance = statistics.mean([qe.relevance_score for qe in query_evaluations])
        avg_accuracy = statistics.mean([qe.accuracy_score for qe in query_evaluations])
        avg_completeness = statistics.mean([qe.completeness_score for qe in query_evaluations])
        avg_response_time = statistics.mean([qe.response_time for qe in query_evaluations])
        avg_context_quality = statistics.mean([qe.context_quality for qe in query_evaluations])
        avg_source_diversity = statistics.mean([qe.source_diversity for qe in query_evaluations])

        # 相关性建议
        if avg_relevance < 0.7:
            recommendations.append("改进检索算法，提高查询与文档的相关性匹配")
            recommendations.append("优化查询理解，更好地识别用户意图")

        # 准确性建议
        if avg_accuracy < 0.75:
            recommendations.append("改进答案生成算法，提高答案准确性")
            recommendations.append("增加事实核查机制")

        # 完整性建议
        if avg_completeness < 0.7:
            recommendations.append("优化上下文构建，提供更全面的信息")
            recommendations.append("改进答案结构，确保信息完整")

        # 响应时间建议
        if avg_response_time > 2.0:
            recommendations.append("优化系统性能，减少响应时间")
            recommendations.append("考虑添加缓存机制")

        # 上下文质量建议
        if avg_context_quality < 0.7:
            recommendations.append("改进文档检索策略，提高上下文质量")
            recommendations.append("优化重排序算法")

        # 来源多样性建议
        if avg_source_diversity < 0.6:
            recommendations.append("增加文档来源的多样性")
            recommendations.append("优化检索策略，避免来源过于集中")

        # 通用建议
        recommendations.append("定期更新知识库，保持信息时效性")
        recommendations.append("收集用户反馈，持续改进系统")

        return recommendations


def create_test_queries() -> List[Dict[str, Any]]:
    """创建测试查询集"""
    return [
        {
            "query": "Python是什么时候创建的？",
            "expected_answer": "Python由Guido van Rossum于1991年创建",
            "category": "factual",
        },
        {
            "query": "机器学习有哪些主要类型？",
            "expected_answer": "机器学习主要包括监督学习、无监督学习和强化学习",
            "category": "factual",
        },
        {
            "query": "数据科学的工作流程包括哪些步骤？",
            "expected_answer": "包括数据收集、清洗、分析、建模和解释",
            "category": "procedural",
        },
        {
            "query": "为什么说深度学习是AI的重要分支？",
            "expected_answer": "因为深度学习在多个领域取得了突破性进展",
            "category": "explanatory",
        },
        {
            "query": "对比Python和其他编程语言的优势",
            "expected_answer": "Python语法简洁、生态丰富、学习曲线平缓",
            "category": "comparative",
        },
    ]


def run_evaluation_demo():
    """运行评估演示"""
    print("=== RetrievalQA 评估演示 ===")

    try:
        # 导入演示模块
        from retrieval_qa_demo import create_sample_documents
        from retrieval_qa_integration import QASystemManager

        # 创建测试数据
        print("1. 创建测试文档...")
        sample_files = create_sample_documents()

        # 创建问答系统
        print("\n2. 创建问答系统...")
        manager = QASystemManager("basic")
        manager.create_system(retriever_type="hybrid")

        # 加载知识库
        print("\n3. 加载知识库...")
        manager.load_knowledge_base(sample_files)

        # 创建评估器
        print("\n4. 创建评估器...")
        evaluator = RetrievalQAEvaluator()

        # 创建测试查询
        test_queries = create_test_queries()

        # 执行评估
        print("\n5. 执行系统评估...")
        evaluation = evaluator.evaluate_system(manager.qa_system, test_queries)

        # 显示结果
        print("\n6. 评估结果:")
        print("=" * 60)
        print(f"总查询数: {evaluation.total_queries}")
        print(f"总体分数: {evaluation.overall_score:.3f}")
        print(f"平均相关性: {evaluation.avg_relevance:.3f}")
        print(f"平均准确性: {evaluation.avg_accuracy:.3f}")
        print(f"平均完整性: {evaluation.avg_completeness:.3f}")
        print(f"平均响应时间: {evaluation.avg_response_time:.3f}秒")
        print(f"上下文质量: {evaluation.avg_context_quality:.3f}")
        print(f"来源多样性: {evaluation.avg_source_diversity:.3f}")

        # 显示详细指标
        print("\n7. 详细指标:")
        print("-" * 40)
        for metric in evaluation.metrics:
            threshold_info = f" (阈值: {metric.threshold})" if metric.threshold else ""
            unit_info = f" {metric.unit}" if metric.unit else ""
            print(f"{metric.name}: {metric.value:.3f}{unit_info}{threshold_info}")

        # 显示优化建议
        print("\n8. 优化建议:")
        print("-" * 40)
        for i, recommendation in enumerate(evaluation.recommendations, 1):
            print(f"{i}. {recommendation}")

        # 显示查询详情
        print("\n9. 查询详情:")
        print("-" * 40)
        for i, qe in enumerate(evaluation.query_evaluations, 1):
            print(f"查询 {i}: {qe.query}")
            print(f"  总分: {qe.overall_score:.3f}")
            print(f"  相关性: {qe.relevance_score:.3f}")
            print(f"  准确性: {qe.accuracy_score:.3f}")
            print(f"  完整性: {qe.completeness_score:.3f}")
            print(f"  响应时间: {qe.response_time:.3f}秒")
            print()

        # 保存评估结果
        print("10. 保存评估结果...")
        result_file = f"evaluation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 转换为可序列化的格式
        evaluation_dict = {
            "total_queries": evaluation.total_queries,
            "avg_relevance": evaluation.avg_relevance,
            "avg_accuracy": evaluation.avg_accuracy,
            "avg_completeness": evaluation.avg_completeness,
            "avg_response_time": evaluation.avg_response_time,
            "avg_context_quality": evaluation.avg_context_quality,
            "avg_source_diversity": evaluation.avg_source_diversity,
            "overall_score": evaluation.overall_score,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "description": metric.description,
                    "threshold": metric.threshold,
                    "unit": metric.unit,
                }
                for metric in evaluation.metrics
            ],
            "recommendations": evaluation.recommendations,
            "query_evaluations": [
                {
                    "query": qe.query,
                    "expected_answer": qe.expected_answer,
                    "actual_answer": qe.actual_answer,
                    "relevance_score": qe.relevance_score,
                    "accuracy_score": qe.accuracy_score,
                    "completeness_score": qe.completeness_score,
                    "response_time": qe.response_time,
                    "context_quality": qe.context_quality,
                    "source_diversity": qe.source_diversity,
                    "overall_score": qe.overall_score,
                }
                for qe in evaluation.query_evaluations
            ],
            "evaluation_time": evaluation.evaluation_time.isoformat(),
        }

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_dict, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存到: {result_file}")

        # 清理
        print("\n11. 清理测试文件...")
        for file_path in sample_files:
            Path(file_path).unlink()

        print("\n评估演示完成！")

    except Exception as e:
        print(f"评估演示出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_evaluation_demo()
