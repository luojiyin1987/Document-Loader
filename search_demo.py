#!/usr/bin/env python3
"""
搜索功能演示 - 展示不同搜索策略的效果
"""

from embeddings import HybridSearch, SimpleEmbeddings, simple_text_search


def demo_search_strategies():
    """演示不同的搜索策略"""

    # 示例文档集合
    documents = [
        "Python是一种高级编程语言，广泛应用于Web开发、数据分析和人工智能",
        "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习",
        "深度学习使用神经网络模拟人脑的学习过程，在图像识别和自然语言处理方面表现出色",
        "自然语言处理(NLP)使计算机能够理解、解释和生成人类语言",
        "计算机视觉技术让机器能够识别和理解图像及视频内容",
        "数据科学结合统计学、机器学习和领域知识来从数据中提取洞察",
        "算法是解决问题的明确步骤，是计算机科学的基础",
        "软件工程涉及软件开发的系统化方法，包括设计、测试和维护",
        "云计算提供按需计算资源，改变了企业的IT架构和部署方式",
        "网络安全保护数字系统和数据免受未授权访问和攻击",
    ]

    queries = [
        "人工智能 机器学习",
        "Python 编程 开发",
        "数据 算法",
        "网络 安全",
        "深度学习 神经网络",
    ]

    print("=" * 80)
    print("搜索策略对比演示")
    print("=" * 80)

    for query in queries:
        print(f"\n查询: '{query}'")
        print("-" * 60)

        # 1. 简单关键词搜索
        print("1. 关键词搜索:")
        keyword_results = simple_text_search(query, documents, top_k=3)
        for result in keyword_results:
            print(f"   分数: {result['score']:.2f} - {result['document'][:50]}...")

        # 2. 语义搜索
        print("\n2. 语义搜索:")
        embedder = SimpleEmbeddings()
        semantic_results = embedder.similarity_search(query, documents, top_k=3)
        for result in semantic_results:
            print(f"   相似度: {result['similarity']:.3f} - {result['document'][:50]}...")

        # 3. 混合搜索
        print("\n3. 混合搜索:")
        hybrid_search = HybridSearch()
        hybrid_results = hybrid_search.search(query, documents, top_k=3)
        for result in hybrid_results:
            print(
                f"   综合分数: {result['combined_score']:.3f} "
                f"(关键词: {result['keyword_score']:.3f}, "
                f"语义: {result['semantic_score']:.3f}) - "
                f"{result['document'][:50]}..."
            )

        print("\n" + "=" * 60)


def performance_comparison():
    """性能对比测试"""
    import time

    # 创建更大的文档集合
    large_documents = [
        f"这是第{i}个文档，内容涉及Python编程、机器学习和数据分析。"
        f"文档编号{i}包含技术信息和实践案例。"
        for i in range(100)
    ]

    queries = ["Python 机器学习", "数据分析 编程", "技术 实践"]

    print("\n性能对比测试")
    print("=" * 60)

    for query in queries:
        print(f"\n查询: '{query}'")

        # 关键词搜索性能
        start_time = time.time()
        for _ in range(10):
            simple_text_search(query, large_documents, top_k=5)
        keyword_time = time.time() - start_time

        # 语义搜索性能
        start_time = time.time()
        embedder = SimpleEmbeddings()
        for _ in range(10):
            embedder.similarity_search(query, large_documents, top_k=5)
        semantic_time = time.time() - start_time

        # 混合搜索性能
        start_time = time.time()
        hybrid_search = HybridSearch()
        for _ in range(10):
            hybrid_search.search(query, large_documents, top_k=5)
        hybrid_time = time.time() - start_time

        print(f"关键词搜索: {keyword_time:.4f}s")
        print(f"语义搜索: {semantic_time:.4f}s")
        print(f"混合搜索: {hybrid_time:.4f}s")


if __name__ == "__main__":
    demo_search_strategies()
    performance_comparison()
