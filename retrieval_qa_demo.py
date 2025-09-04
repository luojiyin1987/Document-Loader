#!/usr/bin/env python3
"""
RetrievalQA 实际使用示例
演示如何使用RetrievalQA系统处理真实文档和回答问题
"""

import sys
from pathlib import Path
from typing import List

# 导入RetrievalQA模块
from retrieval_qa_integration import QASystemManager


def create_sample_documents() -> List[str]:
    """创建示例文档内容"""
    documents = [
        {
            "filename": "ai_intro.txt",
            "content": """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

AI的主要领域包括：

1. 机器学习（Machine Learning）
   - 监督学习：从标记数据中学习模式
   - 无监督学习：从未标记数据中发现结构
   - 强化学习：通过与环境交互学习

2. 深度学习（Deep Learning）
   - 神经网络：模拟人脑神经元结构
   - 卷积神经网络（CNN）：专门处理图像数据
   - 循环神经网络（RNN）：处理序列数据

3. 自然语言处理（NLP）
   - 机器翻译：自动翻译不同语言
   - 情感分析：分析文本情感倾向
   - 文本生成：自动生成人类可读文本

4. 计算机视觉（Computer Vision）
   - 图像识别：识别图像中的对象
   - 人脸识别：识别和验证人脸
   - 目标检测：在图像中定位特定对象

人工智能在医疗、金融、教育、交通等领域都有广泛应用。""",
        },
        {
            "filename": "python_guide.txt",
            "content": """Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python以其简洁的语法和强大的功能而闻名。

Python的主要特点：

1. 简洁易读
   - 使用缩进而非大括号
   - 语法简单直观
   - 代码可读性强

2. 丰富的标准库
   - 内置大量模块和函数
   - 覆盖网络、文件、数据库等操作
   - 开箱即用的功能

3. 第三方生态系统
   - pip包管理器
   - 超过20万个第三方包
   - 活跃的开发者社区

4. 多领域应用
   - Web开发（Django、Flask）
   - 数据科学（NumPy、Pandas）
   - 人工智能（TensorFlow、PyTorch）
   - 自动化脚本

Python的学习曲线相对平缓，适合初学者入门编程。同时，它也是专业人士的首选工具之一。""",
        },
        {
            "filename": "data_science.txt",
            "content": """数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识来从数据中提取洞察。

数据科学工作流程：

1. 数据收集
   - 内部数据库
   - 公开数据集
   - API接口
   - 网络爬虫

2. 数据清洗
   - 处理缺失值
   - 去除重复数据
   - 数据格式转换
   - 异常值检测

3. 探索性数据分析（EDA）
   - 统计摘要
   - 数据可视化
   - 相关性分析
   - 假设检验

4. 特征工程
   - 特征选择
   - 特征转换
   - 特征创建
   - 降维技术

5. 模型构建
   - 机器学习算法
   - 深度学习模型
   - 集成方法
   - 模型调优

6. 结果解释
   - 业务洞察
   - 可视化展示
   - 报告生成
   - 部署上线

常用的工具包括Python（Pandas、NumPy、Scikit-learn）、R语言、SQL等。""",
        },
    ]

    # 保存示例文档到文件
    created_files = []
    for doc in documents:
        file_path = Path(doc["filename"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        created_files.append(str(file_path))

    return created_files


def demo_basic_usage():
    """演示基本使用"""
    print("=== RetrievalQA 基本使用演示 ===")

    # 创建示例文档
    print("1. 创建示例文档...")
    sample_files = create_sample_documents()
    print(f"已创建 {len(sample_files)} 个示例文档")

    # 创建问答系统
    print("\n2. 创建问答系统...")
    manager = QASystemManager("basic")
    manager.create_system(retriever_type="hybrid")

    # 加载知识库
    print("\n3. 加载知识库...")
    manager.load_knowledge_base(sample_files, chunk_size=800, chunk_overlap=150)

    # 测试查询
    test_questions = [
        "Python是什么时候创建的？",
        "机器学习有哪些主要类型？",
        "数据科学的工作流程包括哪些步骤？",
        "深度学习在AI中扮演什么角色？",
        "Python有哪些主要特点？",
    ]

    print("\n4. 执行测试查询...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n查询 {i}: {question}")
        print("-" * 50)

        result = manager.query(question, top_k=3)

        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"来源: {', '.join(result['sources'])}")
        print(f"处理时间: {result['processing_time']:.3f}秒")

        # 显示详细信息
        if result["context_docs_count"] > 0:
            print(f"使用的上下文文档数: {result['context_docs_count']}")

    # 显示系统统计
    print("\n5. 系统统计信息:")
    manager.show_stats()

    # 清理示例文件
    print("\n6. 清理示例文件...")
    for file_path in sample_files:
        Path(file_path).unlink()

    print("\n基本使用演示完成！")


def demo_advanced_usage():
    """演示高级使用"""
    print("\n=== RetrievalQA 高级使用演示 ===")

    # 创建示例文档
    print("1. 创建示例文档...")
    sample_files = create_sample_documents()
    print(f"已创建 {len(sample_files)} 个示例文档")

    # 创建高级问答系统
    print("\n2. 创建高级问答系统...")
    manager = QASystemManager("advanced")
    manager.create_system()

    # 加载知识库
    print("\n3. 加载知识库...")
    manager.load_knowledge_base(
        sample_files, chunk_size=1000, chunk_overlap=200, splitter_type="recursive"
    )

    # 测试不同上下文策略
    strategies = ["relevant_first", "diverse", "comprehensive"]
    test_question = "人工智能有哪些主要应用领域？"

    print(f"\n4. 测试不同上下文策略 (问题: {test_question})")
    print("=" * 60)

    for strategy in strategies:
        print(f"\n策略: {strategy}")
        print("-" * 30)

        result = manager.query(test_question, top_k=3, context_strategy=strategy)

        print(f"答案: {result['answer'][:200]}...")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"上下文分数: {result['context_score']:.3f}")
        print(f"处理时间: {result['processing_time']:.3f}秒")

    # 测试复杂查询
    complex_questions = [
        "对比机器学习和深度学习的区别",
        "分析Python在数据科学中的优势",
        "为什么说数据科学是跨学科领域？",
    ]

    print("\n5. 测试复杂查询...")
    for i, question in enumerate(complex_questions, 1):
        print(f"\n复杂查询 {i}: {question}")
        print("-" * 50)

        result = manager.query(question, top_k=4, context_strategy="comprehensive")

        print(f"答案: {result['answer'][:300]}...")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"来源文档: {len(result['sources'])} 个")
        print(f"上下文Token数: {result['total_tokens']}")

    # 批量查询测试
    print("\n6. 批量查询测试...")
    batch_questions = [
        "什么是监督学习？",
        "Python适合初学者吗？",
        "数据清洗的步骤有哪些？",
    ]

    batch_results = manager.batch_query(batch_questions, top_k=2)

    for i, result in enumerate(batch_results, 1):
        print(f"\n批量查询 {i}: {result['query']}")
        print(f"答案: {result['answer'][:150]}...")
        print(f"置信度: {result['confidence']:.3f}")

    # 显示详细统计
    print("\n7. 高级系统统计信息:")
    manager.show_stats()

    # 清理示例文件
    print("\n8. 清理示例文件...")
    for file_path in sample_files:
        Path(file_path).unlink()

    print("\n高级使用演示完成！")


def demo_web_content():
    """演示网页内容处理"""
    print("\n=== 网页内容处理演示 ===")

    # 使用维基百科的AI相关页面作为示例
    web_urls = [
        "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD",
        "https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0",
    ]

    print("1. 从网页加载内容...")
    print(f"目标URLs: {web_urls}")

    try:
        # 创建问答系统
        manager = QASystemManager("basic")
        manager.create_system(retriever_type="hybrid")

        # 加载网页内容到知识库
        print("\n2. 加载网页内容到知识库...")
        manager.load_knowledge_base(web_urls, chunk_size=1200, chunk_overlap=200)

        # 测试网页内容查询
        web_questions = [
            "人工智能的历史发展如何？",
            "机器学习的主要算法有哪些？",
            "AI在现代社会中有哪些应用？",
        ]

        print("\n3. 执行网页内容查询...")
        for i, question in enumerate(web_questions, 1):
            print(f"\n网页查询 {i}: {question}")
            print("-" * 50)

            result = manager.query(question, top_k=3)

            print(f"答案: {result['answer'][:300]}...")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"来源: {', '.join(result['sources'])}")
            print(f"处理时间: {result['processing_time']:.3f}秒")

        print("\n网页内容处理演示完成！")

    except Exception as e:
        print(f"网页内容处理演示出错: {e}")
        print("这可能是因为网络连接问题或网站访问限制")


def demo_performance_comparison():
    """演示性能比较"""
    print("\n=== 性能比较演示 ===")

    # 创建示例文档
    print("1. 准备测试数据...")
    sample_files = create_sample_documents()

    # 测试不同配置
    configs = [
        {"system": "basic", "retriever": "vector", "name": "基础向量检索"},
        {"system": "basic", "retriever": "hybrid", "name": "基础混合检索"},
        {"system": "advanced", "retriever": "hybrid", "name": "高级混合检索"},
    ]

    test_questions = [
        "Python的特点有哪些？",
        "什么是深度学习？",
        "数据科学的步骤是什么？",
    ]

    results_summary = []

    for config in configs:
        print(f"\n2. 测试配置: {config['name']}")
        print("-" * 40)

        try:
            # 创建系统
            manager = QASystemManager(config["system"])
            manager.create_system(retriever_type=config["retriever"])

            # 加载知识库
            manager.load_knowledge_base(sample_files)

            # 执行测试查询并计时
            import time

            total_time = 0
            total_confidence = 0

            for question in test_questions:
                start_time = time.time()
                result = manager.query(question, top_k=3)
                end_time = time.time()

                total_time += end_time - start_time
                total_confidence += result["confidence"]

            avg_time = total_time / len(test_questions)
            avg_confidence = total_confidence / len(test_questions)

            result_data = {
                "config": config["name"],
                "avg_time": avg_time,
                "avg_confidence": avg_confidence,
                "total_docs": len(sample_files),
            }
            results_summary.append(result_data)

            print(f"平均响应时间: {avg_time:.3f}秒")
            print(f"平均置信度: {avg_confidence:.3f}")

        except Exception as e:
            print(f"配置测试失败: {e}")

    # 显示比较结果
    print("\n3. 性能比较总结:")
    print("=" * 60)
    print(f"{'配置':<15} {'平均时间':<10} {'平均置信度':<12} {'文档数'}")
    print("-" * 60)

    for result in results_summary:
        print(
            f"{result['config']:<15} {result['avg_time']:<10.3f} "
            f"{result['avg_confidence']:<12.3f} {result['total_docs']}"
        )

    # 清理示例文件
    print("\n4. 清理示例文件...")
    for file_path in sample_files:
        Path(file_path).unlink()

    print("\n性能比较演示完成！")


def interactive_demo():
    """交互式演示"""
    print("\n=== 交互式演示 ===")
    print("这个演示将创建一个简单的问答系统，您可以与之交互")

    try:
        # 创建示例文档
        print("1. 创建示例文档...")
        sample_files = create_sample_documents()

        # 创建问答系统
        print("\n2. 创建问答系统...")
        manager = QASystemManager("basic")
        manager.create_system(retriever_type="hybrid")

        # 加载知识库
        print("\n3. 加载知识库...")
        manager.load_knowledge_base(sample_files)

        print("\n4. 进入交互模式...")
        print("您可以询问关于Python、AI、数据科学等问题")
        print("输入 'quit' 退出演示")
        print("=" * 50)

        # 进入交互模式
        manager.interactive_mode()

        # 清理示例文件
        print("\n5. 清理示例文件...")
        for file_path in sample_files:
            Path(file_path).unlink()

    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"交互式演示出错: {e}")


def main():
    """主函数"""
    print("RetrievalQA 使用示例")
    print("=" * 60)

    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()

        if demo_type == "basic":
            demo_basic_usage()
        elif demo_type == "advanced":
            demo_advanced_usage()
        elif demo_type == "web":
            demo_web_content()
        elif demo_type == "performance":
            demo_performance_comparison()
        elif demo_type == "interactive":
            interactive_demo()
        else:
            print("未知演示类型")
            print("可用演示: basic, advanced, web, performance, interactive")
    else:
        # 运行所有演示
        print("运行所有演示...\n")

        try:
            demo_basic_usage()
            demo_advanced_usage()
            demo_performance_comparison()

            print("\n" + "=" * 60)
            print("所有演示完成！")
            print("\n其他可用演示:")
            print("- python retrieval_qa_demo.py web     # 网页内容处理")
            print("- python retrieval_qa_demo.py interactive  # 交互式演示")

        except KeyboardInterrupt:
            print("\n演示被用户中断")
        except Exception as e:
            print(f"演示过程中出错: {e}")


if __name__ == "__main__":
    main()
