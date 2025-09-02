#!/usr/bin/env python3
"""
Test runner script for document loader project
Provides easy-to-use interface for running different types of tests
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_comprehensive import TestSuite
from generate_test_data import TestDataGenerator

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 60)
    
    test_suite = TestSuite()
    report = test_suite.run_all_tests()
    
    return report

def generate_test_data():
    """Generate test data files"""
    print("📁 Generating Test Data")
    print("=" * 60)
    
    generator = TestDataGenerator()
    all_docs = generator.generate_all()
    
    print(f"✅ Generated {len(all_docs)} test documents")
    return all_docs

def run_performance_tests():
    """Run performance-specific tests"""
    print("⚡ Running Performance Tests")
    print("=" * 60)
    
    from test_comprehensive import TestSuite
    
    test_suite = TestSuite()
    test_suite.run_performance_tests()
    
    print("✅ Performance tests completed")

def run_search_tests():
    """Run search-specific tests"""
    print("🔍 Running Search Tests")
    print("=" * 60)
    
    from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search
    
    # Test documents
    documents = [
        "Python是一种高级编程语言，广泛应用于Web开发、数据分析和人工智能",
        "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习",
        "深度学习使用神经网络模拟人脑的学习过程，在图像识别和自然语言处理方面表现出色",
        "自然语言处理(NLP)使计算机能够理解、解释和生成人类语言",
        "计算机视觉技术让机器能够识别和理解图像及视频内容"
    ]
    
    queries = [
        "Python 编程",
        "人工智能 机器学习",
        "深度学习 神经网络",
        "数据 算法",
        "网络 安全"
    ]
    
    print("Testing search strategies...")
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Keyword search
        keyword_results = simple_text_search(query, documents, top_k=3)
        print(f"Keyword search: {len(keyword_results)} results")
        for result in keyword_results:
            print(f"  Score: {result['score']:.2f} - {result['document'][:50]}...")
        
        # Semantic search
        embedder = SimpleEmbeddings()
        semantic_results = embedder.similarity_search(query, documents, top_k=3)
        print(f"Semantic search: {len(semantic_results)} results")
        for result in semantic_results:
            print(f"  Similarity: {result['similarity']:.3f} - {result['document'][:50]}...")
        
        # Hybrid search
        hybrid_search = HybridSearch()
        hybrid_results = hybrid_search.search(query, documents, top_k=3)
        print(f"Hybrid search: {len(hybrid_results)} results")
        for result in hybrid_results:
            print(f"  Combined: {result['combined_score']:.3f} - {result['document'][:50]}...")

def run_splitter_tests():
    """Run text splitter tests"""
    print("✂️ Running Text Splitter Tests")
    print("=" * 60)
    
    from main import CharacterTextSplitter, RecursiveCharacterTextSplitter, StreamingTextSplitter, TokenTextSplitter, SemanticTextSplitter
    
    # Test text
    test_text = """This is a test document for text splitters. It contains multiple paragraphs and sentences that should be split into manageable chunks.

The text splitter should handle various types of content including long paragraphs, short sentences, and different types of separators.

Each splitter implementation has its own strategy for dividing text while preserving context and maintaining readability.

The goal is to create chunks that are small enough to process efficiently but large enough to maintain semantic coherence."""
    
    splitters = [
        ("CharacterTextSplitter", CharacterTextSplitter(chunk_size=100, chunk_overlap=10)),
        ("RecursiveCharacterTextSplitter", RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)),
        ("StreamingTextSplitter", StreamingTextSplitter(chunk_size=100, chunk_overlap=10)),
        ("TokenTextSplitter", TokenTextSplitter(chunk_size=50, chunk_overlap=5)),
        ("SemanticTextSplitter", SemanticTextSplitter(chunk_size=100, chunk_overlap=10))
    ]
    
    for name, splitter in splitters:
        print(f"\n{name}:")
        print("-" * 40)
        chunks = list(splitter.split_text(test_text))
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Chunk sizes: {[len(chunk) for chunk in chunks[:3]]}...")
        if chunks:
            print(f"  First chunk: {chunks[0][:50]}...")
            print(f"  Last chunk: {chunks[-1][:50]}...")

def run_integration_tests():
    """Run integration tests combining multiple components"""
    print("🔗 Running Integration Tests")
    print("=" * 60)
    
    # Generate test data first
    generator = TestDataGenerator()
    all_docs = generator.generate_all()
    
    # Test loading and processing a document
    test_file = Path("test_data/python_programming.txt")
    if test_file.exists():
        print(f"Testing with: {test_file}")
        
        # Load the document
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"  Document size: {len(content)} characters")
        
        # Test text splitting
        from main import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = list(splitter.split_text(content))
        print(f"  Split into {len(chunks)} chunks")
        
        # Test search
        from embeddings import SimpleEmbeddings
        embedder = SimpleEmbeddings()
        results = embedder.similarity_search("Python programming", chunks, top_k=3)
        print(f"  Search results: {len(results)} matches")
        
        print("  ✅ Integration test completed")
    else:
        print("  ❌ Test file not found")

def run_memory_tests():
    """Run memory efficiency tests"""
    print("🧠 Running Memory Efficiency Tests")
    print("=" * 60)
    
    from main import StreamingTextSplitter
    
    # Create a large document
    large_text = "This is a test sentence for memory efficiency testing. " * 10000
    print(f"  Test document size: {len(large_text)} characters")
    
    # Test streaming splitter
    splitter = StreamingTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = list(splitter.split_text(large_text))
    print(f"  Streaming splitter: {len(chunks)} chunks")
    
    # Verify chunk sizes
    max_chunk_size = max(len(chunk) for chunk in chunks)
    print(f"  Maximum chunk size: {max_chunk_size} characters")
    
    if max_chunk_size <= 1000:
        print("  ✅ Memory efficiency test passed")
    else:
        print("  ❌ Memory efficiency test failed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test runner for document loader project")
    parser.add_argument("--test-type", choices=["comprehensive", "generate-data", "performance", "search", "splitter", "integration", "memory", "all"], 
                       default="all", help="Type of test to run")
    
    args = parser.parse_args()
    
    if args.test_type == "comprehensive":
        run_comprehensive_tests()
    elif args.test_type == "generate-data":
        generate_test_data()
    elif args.test_type == "performance":
        run_performance_tests()
    elif args.test_type == "search":
        run_search_tests()
    elif args.test_type == "splitter":
        run_splitter_tests()
    elif args.test_type == "integration":
        run_integration_tests()
    elif args.test_type == "memory":
        run_memory_tests()
    elif args.test_type == "all":
        print("🚀 Running All Tests")
        print("=" * 60)
        
        # Generate test data first
        generate_test_data()
        
        # Run all test types
        run_splitter_tests()
        run_search_tests()
        run_memory_tests()
        run_integration_tests()
        run_performance_tests()
        run_comprehensive_tests()
        
        print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()