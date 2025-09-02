#!/usr/bin/env python3
"""
向量存储使用示例 - 展示如何将向量存储集成到文档处理流程中
"""

# ===== 项目自定义模块导入 =====
from main import read_txt_file
from text_splitter import create_text_splitter
from vector_store_integration import create_vector_store_integration


def demo_vector_store_workflow():
    """演示向量存储工作流程"""
    print("=== 向量存储工作流程演示 ===")

    # 示例文档内容
    sample_content = """
    人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是人工智能的核心技术，它使计算机能够从数据中学习而无需明确编程。

    深度学习是机器学习的一个子领域，使用人工神经网络来模拟人脑的工作方式。
    神经网络由相互连接的节点层组成，能够识别数据中的复杂模式。

    自然语言处理（NLP）是AI的另一个重要分支，专注于计算机与人类语言之间的交互。
    NLP技术包括机器翻译、情感分析、文本摘要和问答系统等。

    计算机视觉使机器能够理解和解释视觉信息，在图像识别、物体检测和自动驾驶等领域有广泛应用。
    强化学习通过试错来学习，在游戏、机器人和优化问题中表现出色。
    """

    print("1. 文本分割")
    print("-" * 40)

    # 创建文本分割器
    splitter = create_text_splitter(splitter_type="recursive", chunk_size=200, chunk_overlap=50)

    # 分割文档
    chunks = splitter.create_documents(
        sample_content, {"source": "demo_content.txt", "type": "ai_introduction"}
    )

    print(f"文档已分割为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
        print(f"块 {i+1}: {chunk['page_content'][:100]}...")

    print("\n2. 添加到向量存储")
    print("-" * 40)

    # 添加到向量存储
    integration = create_vector_store_integration("demo_vector_store.pkl")
    doc_ids = integration.add_documents_from_chunks(chunks)
    print(f"已添加 {len(doc_ids)} 个文档到向量存储")

    # 保存向量存储（使用安全的JSON格式）
    integration.save_vector_store()
    print("向量存储已保存（JSON格式）")

    print("\n3. 搜索测试")
    print("-" * 40)

    # 搜索测试
    queries = [
        "机器学习是什么",
        "深度学习和神经网络",
        "自然语言处理应用",
        "计算机视觉技术",
        "强化学习原理",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = integration.search_similar_chunks(query, top_k=2)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  结果 {i}: 相似度={result.get('similarity', 'N/A'):.3f}")
                print(f"    内容: {result['document']['page_content'][:80]}...")
                print(f"    来源: {result['document']['metadata'].get('source', 'Unknown')}")
        else:
            print("  未找到相关结果")

    print("\n4. 向量存储统计")
    print("-" * 40)

    # 获取统计信息
    if integration.load_vector_store():
        stats = integration.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("无法加载向量存储文件")

    print("\n演示完成！")


def demo_file_processing_workflow():
    """演示文件处理工作流程"""
    print("\n=== 文件处理工作流程演示 ===")

    # 这里可以处理实际的文件
    # 由于是演示，我们使用示例内容
    sample_file_path = "sample_document.txt"

    # 创建示例文件
    sample_content = """
    Python编程语言特点：
    1. 简单易学：语法清晰，适合初学者
    2. 面向对象：支持面向对象编程
    3. 可移植性：跨平台运行
    4. 丰富的库：拥有大量的第三方库
    5. 社区支持：活跃的开发者社区

    Python应用领域：
    - Web开发：Django、Flask框架
    - 数据科学：Pandas、NumPy、Matplotlib
    - 人工智能：TensorFlow、PyTorch
    - 自动化脚本：系统管理、测试
    - 桌面应用：Tkinter、PyQt
    """

    # 写入示例文件
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    print(f"创建了示例文件: {sample_file_path}")

    # 读取文件
    content = read_txt_file(sample_file_path)
    print(f"读取文件内容，长度: {len(content)} 字符")

    # 分割文件
    splitter = create_text_splitter("recursive", 150, 30)
    chunks = splitter.create_documents(content, {"source": sample_file_path})

    print(f"文件分割为 {len(chunks)} 个块")

    # 添加到向量存储
    integration = create_vector_store_integration("file_vector_store.pkl")
    doc_ids = integration.add_documents_from_chunks(chunks)

    print(f"添加了 {len(doc_ids)} 个文档到向量存储")

    # 搜索测试
    print("\n搜索测试:")
    query = "Python 应用领域"
    results = integration.search_similar_chunks(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"  {i}. 相似度: {result.get('similarity', 'N/A'):.3f}")
        print(f"     {result['document']['page_content'][:100]}...")

    # 保存向量存储（使用安全的JSON格式）
    integration.save_vector_store()
    print("\n向量存储已保存到: file_vector_store.json")

    # 清理示例文件
    import os

    os.remove(sample_file_path)
    print(f"清理示例文件: {sample_file_path}")


if __name__ == "__main__":
    demo_vector_store_workflow()
    demo_file_processing_workflow()
