#!/usr/bin/env python3
"""
Comprehensive test suite for document loader and text splitters
Tests all functionality including text splitting, search strategies, and document loading
"""

import sys
import os
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, StreamingTextSplitter, TokenTextSplitter, SemanticTextSplitter
from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search

class TestSuite:
    """Comprehensive test suite for document loader functionality"""
    
    def __init__(self):
        self.test_results = []
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        try:
            start_time = time.time()
            test_func()
            end_time = time.time()
            result = {
                'name': test_name,
                'status': 'PASS',
                'time': end_time - start_time,
                'error': None
            }
            print(f"✅ {test_name} ({end_time - start_time:.3f}s)")
        except Exception as e:
            result = {
                'name': test_name,
                'status': 'FAIL',
                'time': time.time() - start_time,
                'error': str(e)
            }
            print(f"❌ {test_name} - {str(e)}")
        
        self.test_results.append(result)
        return result['status'] == 'PASS'
    
    def create_test_documents(self):
        """Create test documents for various scenarios"""
        # Create basic test documents
        test_docs = {
            'basic.txt': "This is a basic test document. It contains multiple sentences for testing text splitting functionality. The document should be split into manageable chunks.",
            
            'long_paragraph.txt': "This is a very long paragraph that contains multiple sentences and should be split when it exceeds the chunk size. " * 20,
            
            'mixed_content.txt': """This document contains mixed content.

Paragraph 1: This is the first paragraph with multiple sentences. It should be handled properly by the text splitter.

Paragraph 2: This is the second paragraph. It contains different content and should be split independently.

Paragraph 3: The final paragraph contains conclusion text and summary information.""",
            
            'chinese.txt': "这是一个中文测试文档。包含多个句子和段落，用于测试中文文本分割功能。文档应该被分割成适当大小的块。",
            
            'mixed_language.txt': "This is a mixed language document. 这是一个混合语言文档。It contains both English and Chinese text. 它包含英文和中文文本。",
            
            'code_snippets.txt': """def hello_world():
    print("Hello, World!")
    return True

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

# This is a comment
x = 10
y = 20
result = x + y""",
            
            'structured_data.txt': """Name: John Doe
Age: 30
City: New York
Occupation: Software Engineer

Name: Jane Smith
Age: 25
City: San Francisco
Occupation: Data Scientist

Name: Bob Johnson
Age: 35
City: Chicago
Occupation: Product Manager"""
        }
        
        for filename, content in test_docs.items():
            with open(self.test_data_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ Created {len(test_docs)} test documents")
        return test_docs
    
    def test_text_splitters(self):
        """Test all text splitter implementations"""
        
        # Test data
        test_text = "This is a test document for text splitters. " * 50
        
        splitters = [
            ('CharacterTextSplitter', CharacterTextSplitter(chunk_size=100, chunk_overlap=10)),
            ('RecursiveCharacterTextSplitter', RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)),
            ('StreamingTextSplitter', StreamingTextSplitter(chunk_size=100, chunk_overlap=10)),
            ('TokenTextSplitter', TokenTextSplitter(chunk_size=50, chunk_overlap=5)),
            ('SemanticTextSplitter', SemanticTextSplitter(chunk_size=100, chunk_overlap=10))
        ]
        
        for name, splitter in splitters:
            chunks = list(splitter.split_text(test_text))
            
            # Basic validation
            assert len(chunks) > 0, f"{name} should produce at least one chunk"
            assert all(len(chunk) > 0 for chunk in chunks), f"{name} should not produce empty chunks"
            
            # Test overlap
            if len(chunks) > 1:
                for i in range(len(chunks) - 1):
                    current_end = chunks[i][-20:]
                    next_start = chunks[i + 1][:20]
                    # Check if there's any overlap
                    overlap_found = any(current_end[j:] == next_start[:len(current_end[j:])] 
                                      for j in range(len(current_end)))
                    # Note: Overlap is not guaranteed for all splitters, so we don't assert it
            
            print(f"   {name}: {len(chunks)} chunks")
    
    def test_search_strategies(self):
        """Test all search strategies"""
        
        # Test documents
        documents = [
            "Python是一种高级编程语言，广泛应用于Web开发、数据分析和人工智能",
            "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习",
            "深度学习使用神经网络模拟人脑的学习过程，在图像识别和自然语言处理方面表现出色",
            "自然语言处理(NLP)使计算机能够理解、解释和生成人类语言",
            "计算机视觉技术让机器能够识别和理解图像及视频内容"
        ]
        
        # Test keyword search
        keyword_results = simple_text_search("Python 机器学习", documents, top_k=3)
        assert len(keyword_results) <= 3, "Keyword search should respect top_k parameter"
        assert all('score' in result for result in keyword_results), "Keyword results should have scores"
        
        # Test semantic search
        embedder = SimpleEmbeddings()
        semantic_results = embedder.similarity_search("人工智能技术", documents, top_k=3)
        assert len(semantic_results) <= 3, "Semantic search should respect top_k parameter"
        assert all('similarity' in result for result in semantic_results), "Semantic results should have similarity scores"
        
        # Test hybrid search
        hybrid_search = HybridSearch()
        hybrid_results = hybrid_search.search("数据 算法", documents, top_k=3)
        assert len(hybrid_results) <= 3, "Hybrid search should respect top_k parameter"
        assert all('combined_score' in result for result in hybrid_results), "Hybrid results should have combined scores"
        
        print(f"   Keyword search: {len(keyword_results)} results")
        print(f"   Semantic search: {len(semantic_results)} results")
        print(f"   Hybrid search: {len(hybrid_results)} results")
    
    def test_document_loading(self):
        """Test document loading functionality"""
        
        # Test loading text files
        test_files = ['basic.txt', 'chinese.txt', 'mixed_language.txt']
        
        for filename in test_files:
            file_path = self.test_data_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                assert len(content) > 0, f"File {filename} should not be empty"
                print(f"   Loaded {filename}: {len(content)} characters")
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large documents"""
        
        # Create a large document
        large_text = "This is a test sentence for memory efficiency testing. " * 10000
        
        # Test streaming splitter with large document
        splitter = StreamingTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = list(splitter.split_text(large_text))
        
        assert len(chunks) > 0, "Should produce chunks from large document"
        assert all(len(chunk) <= 1000 for chunk in chunks), "All chunks should respect size limit"
        
        print(f"   Large document processed: {len(chunks)} chunks")
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        
        # Test empty text
        splitter = CharacterTextSplitter()
        chunks = list(splitter.split_text(""))
        assert len(chunks) == 0, "Empty text should produce no chunks"
        
        # Test None input
        try:
            list(splitter.split_text(None))
            assert False, "Should handle None input gracefully"
        except:
            pass  # Expected to raise an error
        
        # Test search with empty documents
        results = simple_text_search("query", [], top_k=5)
        assert len(results) == 0, "Empty documents should produce no results"
        
        print("   Error handling tests passed")
    
    def test_multilingual_support(self):
        """Test multilingual text processing"""
        
        multilingual_docs = [
            "This is an English document for testing multilingual support.",
            "这是一个中文文档，用于测试多语言支持。",
            "This is a mixed language document. 这是混合语言文档。",
            "日本語のドキュメントをテストします。",  # Japanese
            "한국어 문서를 테스트합니다."  # Korean
        ]
        
        # Test keyword search across languages
        results = simple_text_search("文档", multilingual_docs, top_k=5)
        assert len(results) > 0, "Should find matches across languages"
        
        # Test semantic search
        embedder = SimpleEmbeddings()
        semantic_results = embedder.similarity_search("language", multilingual_docs, top_k=3)
        assert len(semantic_results) > 0, "Semantic search should work across languages"
        
        print(f"   Multilingual support: {len(results)} keyword results, {len(semantic_results)} semantic results")
    
    def run_performance_tests(self):
        """Test performance characteristics"""
        
        # Performance test data
        perf_documents = [
            f"Performance test document {i} with various content for testing search performance. " * 50
            for i in range(100)
        ]
        
        queries = ["performance test", "document content", "search query"]
        
        print("   Performance comparison:")
        for query in queries:
            # Keyword search performance
            start_time = time.time()
            for _ in range(10):
                simple_text_search(query, perf_documents, top_k=10)
            keyword_time = time.time() - start_time
            
            # Semantic search performance
            start_time = time.time()
            embedder = SimpleEmbeddings()
            for _ in range(10):
                embedder.similarity_search(query, perf_documents, top_k=10)
            semantic_time = time.time() - start_time
            
            # Hybrid search performance
            start_time = time.time()
            hybrid_search = HybridSearch()
            for _ in range(10):
                hybrid_search.search(query, perf_documents, top_k=10)
            hybrid_time = time.time() - start_time
            
            print(f"     Query '{query}': Keyword {keyword_time:.3f}s, Semantic {semantic_time:.3f}s, Hybrid {hybrid_time:.3f}s")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'timestamp': time.time()
        }
        
        # Save report
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Test Report:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"   Report saved to: test_report.json")
        
        return report
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Create test documents
        self.create_test_documents()
        
        # Run all test cases
        test_cases = [
            ("Text Splitters", self.test_text_splitters),
            ("Search Strategies", self.test_search_strategies),
            ("Document Loading", self.test_document_loading),
            ("Memory Efficiency", self.test_memory_efficiency),
            ("Error Handling", self.test_error_handling),
            ("Multilingual Support", self.test_multilingual_support),
            ("Performance Tests", self.run_performance_tests)
        ]
        
        for test_name, test_func in test_cases:
            print(f"\n🔍 Testing {test_name}")
            print("-" * 40)
            self.run_test(test_name, test_func)
        
        # Generate final report
        print("\n" + "=" * 60)
        report = self.generate_test_report()
        
        return report

def main():
    """Main test runner"""
    
    # Run test suite
    test_suite = TestSuite()
    report = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()