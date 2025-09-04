#!/usr/bin/env python3
"""
搜索引擎工具模块 - 提供多种搜索引擎接口
支持网络搜索、API调用和结果处理功能
"""

# ===== 标准库导入 =====
import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class SearchEngineBase:
    """搜索引擎基类"""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.name = self.__class__.__name__

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """搜索方法，子类需要实现"""
        raise NotImplementedError("子类必须实现 search 方法")

    def _make_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """发送HTTP请求"""
        try:
            request = Request(url, headers=headers or {})
            with urlopen(request, timeout=self.timeout) as response:  # nosec B310
                return response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            raise Exception(f"请求失败: {e}")


class WebSearchEngine(SearchEngineBase):
    """通用网络搜索引擎 - 使用DuckDuckGo API"""

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """使用DuckDuckGo进行网络搜索"""
        try:
            # DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"

            response_text = self._make_request(url)
            data = json.loads(response_text)

            results = []

            # 处理主要结果
            if data.get("AbstractText"):
                results.append(
                    {
                        "title": data.get("Heading", ""),
                        "snippet": data.get("AbstractText", ""),
                        "url": data.get("AbstractURL", ""),
                        "source": "DuckDuckGo",
                        "type": "featured",
                    }
                )

            # 处理相关主题
            if data.get("RelatedTopics"):
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(
                            {
                                "title": topic.get("FirstURL", "")
                                .split("/")[-1]
                                .replace("_", " "),
                                "snippet": topic.get("Text", ""),
                                "url": topic.get("FirstURL", ""),
                                "source": "DuckDuckGo",
                                "type": "related",
                            }
                        )

            # 如果没有找到结果，尝试英文搜索
            if not results and self._contains_chinese(query):
                english_query = self._translate_to_english(query)
                if english_query != query:
                    print(f"尝试英文搜索: {english_query}")
                    return self.search(english_query, num_results)

            return results[:num_results]

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def _contains_chinese(self, text: str) -> bool:
        """检查是否包含中文字符"""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _translate_to_english(self, text: str) -> str:
        """简单的中文到英文翻译映射"""
        translations = {
            "编程": "programming",
            "教程": "tutorial",
            "学习": "learn",
            "开发": "development",
            "代码": "code",
            "软件": "software",
            "应用": "application",
            "系统": "system",
            "数据": "data",
            "算法": "algorithm",
            "人工智能": "artificial intelligence",
            "机器学习": "machine learning",
            "深度学习": "deep learning",
            "Python": "Python",
            "Java": "Java",
            "JavaScript": "JavaScript",
        }

        result = text
        for chinese, english in translations.items():
            result = result.replace(chinese, f" {english} ")

        return result.strip()


class BingSearchEngine(SearchEngineBase):
    """Bing搜索引擎 - 需要API密钥"""

    def __init__(self, api_key: str, timeout: int = 10):
        super().__init__(api_key, timeout)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """使用Bing搜索API进行搜索"""
        if not self.api_key:
            raise ValueError("Bing搜索需要API密钥")

        try:
            params = {
                "q": query,
                "count": num_results,
                "mkt": "zh-CN",
                "safesearch": "Moderate",
            }

            url = f"{self.base_url}?{urlencode(params)}"
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            response_text = self._make_request(url, headers)
            data = json.loads(response_text)

            results = []

            # 处理网页搜索结果
            if "webPages" in data and "value" in data["webPages"]:
                for item in data["webPages"]["value"]:
                    results.append(
                        {
                            "title": item.get("name", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("url", ""),
                            "source": "Bing",
                            "type": "web",
                            "display_url": item.get("displayUrl", ""),
                        }
                    )

            return results[:num_results]

        except Exception as e:
            print(f"Bing搜索失败: {e}")
            return []


class SerpApiSearchEngine(SearchEngineBase):
    """SerpApi搜索引擎 - 需要API密钥"""

    def __init__(self, api_key: str, timeout: int = 10):
        super().__init__(api_key, timeout)
        self.base_url = "https://serpapi.com/search"

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """使用SerpApi进行搜索"""
        if not self.api_key:
            raise ValueError("SerpApi需要API密钥")

        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "hl": "zh-cn",
                "gl": "cn",
            }

            url = f"{self.base_url}?{urlencode(params)}"
            response_text = self._make_request(url)
            data = json.loads(response_text)

            results = []

            # 处理有机搜索结果
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "source": "SerpApi",
                            "type": "organic",
                            "display_url": item.get("displayed_link", ""),
                            "position": item.get("position", 0),
                        }
                    )

            return results[:num_results]

        except Exception as e:
            print(f"SerpApi搜索失败: {e}")
            return []


class GoogleSearchEngine(SearchEngineBase):
    """Google搜索引擎 - 模拟搜索（需要处理验证码）"""

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """模拟Google搜索（简化版本）"""
        try:
            # 注意：实际的Google搜索需要处理验证码和反爬机制
            # 这里提供一个简化的实现

            url = f"https://www.google.com/search?q={quote(query)}&num={num_results}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }

            response_text = self._make_request(url, headers)
            print(response_text)

            # 简单的HTML解析（实际应用中建议使用BeautifulSoup）
            results: List[Dict[str, Any]] = []

            # 这里应该实现更复杂的HTML解析
            # 由于Google的反爬机制，这个实现可能不稳定

            return results[:num_results]

        except Exception as e:
            print(f"Google搜索失败: {e}")
            return []


class SearchEngineManager:
    """搜索引擎管理器 - 统一管理多个搜索引擎"""

    def __init__(self):
        self.engines: Dict[str, SearchEngineBase] = {}
        self.default_engine = "web"

    def register_engine(self, name: str, engine: SearchEngineBase):
        """注册搜索引擎"""
        self.engines[name] = engine
        print(f"已注册搜索引擎: {name}")

    def set_default_engine(self, name: str):
        """设置默认搜索引擎"""
        if name in self.engines:
            self.default_engine = name
            print(f"默认搜索引擎设置为: {name}")
        else:
            print(f"搜索引擎 {name} 不存在")

    def search(
        self, query: str, engine: Optional[str] = None, num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """使用指定搜索引擎进行搜索"""
        engine_name = engine or self.default_engine

        if engine_name not in self.engines:
            print(f"搜索引擎 {engine_name} 不存在，使用默认引擎")
            engine_name = self.default_engine

        search_engine = self.engines[engine_name]

        print(f"使用 {engine_name} 搜索: {query}")
        results = search_engine.search(query, num_results)

        print(f"找到 {len(results)} 个结果")
        return results

    def list_engines(self) -> List[str]:
        """列出所有可用的搜索引擎"""
        return list(self.engines.keys())

    def multi_search(
        self, query: str, engines: List[str], num_results: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """多引擎搜索"""
        results = {}

        for engine_name in engines:
            if engine_name in self.engines:
                try:
                    engine_results = self.engines[engine_name].search(
                        query, num_results
                    )
                    results[engine_name] = engine_results
                except Exception as e:
                    print(f"搜索引擎 {engine_name} 搜索失败: {e}")
                    results[engine_name] = []

        return results


# 工厂函数
def create_search_engine_manager() -> SearchEngineManager:
    """创建并配置搜索引擎管理器"""
    manager = SearchEngineManager()

    # 注册默认的Web搜索引擎
    web_engine = WebSearchEngine()
    manager.register_engine("web", web_engine)
    manager.register_engine("duckduckgo", web_engine)

    return manager


def create_bing_engine(api_key: str) -> BingSearchEngine:
    """创建Bing搜索引擎"""
    return BingSearchEngine(api_key)


def create_serpapi_engine(api_key: str) -> SerpApiSearchEngine:
    """创建SerpApi搜索引擎"""
    return SerpApiSearchEngine(api_key)


def format_search_results(
    results: List[Dict[str, Any]], show_snippet: bool = True
) -> str:
    """格式化搜索结果"""
    if not results:
        return "没有找到搜索结果"

    formatted_output = []
    formatted_output.append(f"找到 {len(results)} 个搜索结果:")
    formatted_output.append("=" * 50)

    for i, result in enumerate(results, 1):
        formatted_output.append(f"结果 {i}:")
        formatted_output.append(f"标题: {result.get('title', '无标题')}")
        formatted_output.append(f"链接: {result.get('url', '')}")

        if show_snippet and result.get("snippet"):
            formatted_output.append(f"摘要: {result['snippet']}")

        if result.get("display_url"):
            formatted_output.append(f"显示链接: {result['display_url']}")

        formatted_output.append(f"来源: {result.get('source', '未知')}")
        formatted_output.append("-" * 30)

    return "\n".join(formatted_output)


# 使用示例
if __name__ == "__main__":
    # 创建搜索引擎管理器
    manager = create_search_engine_manager()

    # 测试搜索
    print("=== 搜索引擎测试 ===")

    # Web搜索
    query = "Python 编程教程"
    results = manager.search(query, "web", 3)
    print(format_search_results(results))

    # 如果有API密钥，可以测试其他搜索引擎
    # bing_engine = create_bing_engine("your-api-key")
    # manager.register_engine('bing', bing_engine)

    # serpapi_engine = create_serpapi_engine("your-api-key")
    # manager.register_engine('serpapi', serpapi_engine)
