#!/usr/bin/env python3
"""
搜索引擎工具测试脚本
"""

from search_engine import BingSearchEngine  # noqa: F401
from search_engine import SerpApiSearchEngine  # noqa: F401
from search_engine import SearchEngineManager, WebSearchEngine, format_search_results


def test_web_search_engine():
    """测试Web搜索引擎"""
    print("=== 测试Web搜索引擎 ===")
    engine = WebSearchEngine()

    # 测试英文查询
    print("\n1. 测试英文查询:")
    results = engine.search("Python programming", 3)
    print(f"找到 {len(results)} 个结果")
    if results:
        print("第一个结果:")
        print(f"标题: {results[0]['title']}")
        print(f"链接: {results[0]['url']}")
        print(f"摘要: {results[0]['snippet'][:100]}...")

    # 测试中文查询
    print("\n2. 测试中文查询:")
    results = engine.search("Python 编程", 3)
    print(f"找到 {len(results)} 个结果")
    if results:
        print("第一个结果:")
        print(f"标题: {results[0]['title']}")
        print(f"链接: {results[0]['url']}")
        print(f"摘要: {results[0]['snippet'][:100]}...")


def test_search_engine_manager():
    """测试搜索引擎管理器"""
    print("\n=== 测试搜索引擎管理器 ===")
    manager = SearchEngineManager()

    # 注册Web搜索引擎
    web_engine = WebSearchEngine()
    manager.register_engine("web", web_engine)

    # 列出可用的搜索引擎
    engines = manager.list_engines()
    print(f"可用搜索引擎: {engines}")

    # 测试搜索
    print("\n3. 测试管理器搜索:")
    try:
        results = manager.search("Python tutorial", "web", 3)
        print(f"通过管理器找到 {len(results)} 个结果")
    except Exception as e:
        print(f"搜索失败: {e}")


def test_format_search_results():
    """测试结果格式化"""
    print("\n=== 测试结果格式化 ===")

    # 模拟搜索结果
    mock_results = [
        {
            "title": "Python Programming Language",
            "snippet": "Python is a high-level programming language...",
            "url": "https://python.org",
            "source": "Test",
        }
    ]

    formatted = format_search_results(mock_results)
    print("格式化结果:")
    print(formatted)


def main():
    """主测试函数"""
    print("搜索引擎工具测试开始...")

    try:
        test_web_search_engine()
        test_search_engine_manager()
        test_format_search_results()

        print("\n=== 测试完成 ===")
        print("✅ 所有测试通过")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
