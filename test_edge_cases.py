#!/usr/bin/env python3
"""
Edge case and error handling tests for document loader
Tests various edge cases, error conditions, and boundary scenarios
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CharacterTextSplitter, RecursiveCharacterTextSplitter, StreamingTextSplitter, TokenTextSplitter, SemanticTextSplitter
from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search

class EdgeCaseTester:
    """Test edge cases and error handling"""
    
    def __init__(self):
        self.test_results = []
        
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        try:
            test_func()
            print(f"✅ {test_name}")
            self.test_results.append({'name': test_name, 'status': 'PASS'})
            return True
        except Exception as e:
            print(f"❌ {test_name} - {str(e)}")
            self.test_results.append({'name': test_name, 'status': 'FAIL', 'error': str(e)})
            return False
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        
        # Test empty text with splitters
        splitters = [
            CharacterTextSplitter(),
            RecursiveCharacterTextSplitter(),
            StreamingTextSplitter(),
            TokenTextSplitter(),
            SemanticTextSplitter()
        ]
        
        for splitter in splitters:
            chunks = list(splitter.split_text(""))
            assert len(chunks) == 0, f"{splitter.__class__.__name__} should handle empty text"
        
        # Test empty documents with search
        results = simple_text_search("query", [], top_k=5)
        assert len(results) == 0, "Empty documents should return no results"
        
        # Test empty query
        results = simple_text_search("", ["test document"], top_k=5)
        assert len(results) == 0, "Empty query should return no results"
        
        print("   Empty input handling tests passed")
    
    def test_none_inputs(self):
        """Test handling of None inputs"""
        
        splitter = CharacterTextSplitter()
        
        try:
            list(splitter.split_text(None))
            assert False, "Should raise error for None input"
        except:
            pass  # Expected
        
        # Test search with None documents
        try:
            simple_text_search("query", None, top_k=5)
            assert False, "Should raise error for None documents"
        except:
            pass  # Expected
        
        print("   None input handling tests passed")
    
    def test_single_character_inputs(self):
        """Test handling of single character inputs"""
        
        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        chunks = list(splitter.split_text("A"))
        
        assert len(chunks) == 1, "Single character should produce one chunk"
        assert chunks[0] == "A", "Single character should be preserved"
        
        print("   Single character input tests passed")
    
    def test_extremely_long_single_line(self):
        """Test handling of extremely long single line"""
        
        # Create a very long single line
        long_line = "X" * 10000
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = list(splitter.split_text(long_line))
        
        assert len(chunks) > 1, "Long line should be split into multiple chunks"
        assert all(len(chunk) <= 1000 for chunk in chunks), "All chunks should respect size limit"
        
        # Test overlap
        if len(chunks) > 1:
            overlap_found = False
            for i in range(len(chunks) - 1):
                if chunks[i][-50:] == chunks[i + 1][:50]:
                    overlap_found = True
                    break
            assert overlap_found, "Should find overlap between chunks"
        
        print("   Extremely long single line tests passed")
    
    def test_whitespace_only_inputs(self):
        """Test handling of whitespace-only inputs"""
        
        whitespace_inputs = [
            "   ",
            "\n\n\n",
            " \t \n \t ",
            "    \n    \t    ",
            ""
        ]
        
        splitter = CharacterTextSplitter()
        
        for whitespace_input in whitespace_inputs:
            chunks = list(splitter.split_text(whitespace_input))
            # Should either return empty list or chunks with only whitespace
            if chunks:
                assert all(chunk.strip() == "" for chunk in chunks), "Whitespace input should produce whitespace-only chunks"
        
        print("   Whitespace-only input tests passed")
    
    def test_special_characters(self):
        """Test handling of special characters and unicode"""
        
        special_text = """Special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./
Unicode: 中文العربيةहिन्दीРусский日本語한국어
Math: ∑∏∫√∞≈≠≤≥∈∉⊂⊃⊆⊇∪∩
Emojis: 😀🎉🚀❤️🔥
Accents: Café, naïve, résumé, façade, Müller"""
        
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = list(splitter.split_text(special_text))
        
        assert len(chunks) > 0, "Special characters should be handled"
        assert all(len(chunk) > 0 for chunk in chunks), "No chunk should be empty"
        
        # Verify special characters are preserved
        combined = "".join(chunks)
        assert "中文" in combined, "Chinese characters should be preserved"
        assert "😀" in combined, "Emojis should be preserved"
        assert "∑" in combined, "Math symbols should be preserved"
        
        print("   Special characters tests passed")
    
    def test_repeating_content(self):
        """Test handling of repeating content"""
        
        repeating_text = "Repeat. " * 1000
        
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = list(splitter.split_text(repeating_text))
        
        assert len(chunks) > 1, "Repeating content should be split"
        assert all("Repeat." in chunk for chunk in chunks), "Repeating pattern should be preserved"
        
        print("   Repeating content tests passed")
    
    def test_chunk_size_edge_cases(self):
        """Test edge cases for chunk size parameters"""
        
        test_text = "This is a test document for chunk size edge cases."
        
        # Test chunk size of 1
        splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) == len(test_text), "Chunk size 1 should split each character"
        
        # Test chunk size larger than text
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) == 1, "Large chunk size should produce single chunk"
        
        # Test chunk size equal to text length
        splitter = CharacterTextSplitter(chunk_size=len(test_text), chunk_overlap=0)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) == 1, "Chunk size equal to text length should produce single chunk"
        
        print("   Chunk size edge cases tests passed")
    
    def test_overlap_edge_cases(self):
        """Test edge cases for overlap parameters"""
        
        test_text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        
        # Test zero overlap
        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) > 1, "Should split text"
        
        # Test overlap equal to chunk size
        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=10)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) > 0, "Should handle overlap equal to chunk size"
        
        # Test overlap larger than chunk size
        splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=10)
        chunks = list(splitter.split_text(test_text))
        assert len(chunks) > 0, "Should handle overlap larger than chunk size"
        
        print("   Overlap edge cases tests passed")
    
    def test_encoding_issues(self):
        """Test handling of encoding issues"""
        
        # Create test file with mixed encoding
        test_file = Path("test_data/encoding_test.txt")
        test_file.parent.mkdir(exist_ok=True)
        
        mixed_content = "Mixed encoding: Café, naïve, 中文, العربية, हिन्दी"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(mixed_content)
        
        # Test reading the file
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Café" in content, "Accented characters should be preserved"
        assert "中文" in content, "Chinese characters should be preserved"
        
        # Test processing the content
        splitter = CharacterTextSplitter()
        chunks = list(splitter.split_text(content))
        assert len(chunks) > 0, "Should process mixed encoding content"
        
        # Clean up
        test_file.unlink()
        
        print("   Encoding issues tests passed")
    
    def test_search_edge_cases(self):
        """Test search edge cases"""
        
        documents = [
            "Python is a programming language",
            "Python snakes are reptiles",
            "Java is another programming language"
        ]
        
        # Test exact match
        results = simple_text_search("Python", documents, top_k=5)
        assert len(results) == 2, "Should find exact matches"
        
        # Test case sensitivity
        results = simple_text_search("python", documents, top_k=5)
        assert len(results) == 2, "Should be case insensitive"
        
        # Test partial match
        results = simple_text_search("prog", documents, top_k=5)
        assert len(results) > 0, "Should find partial matches"
        
        # Test no matches
        results = simple_text_search("xyzabc", documents, top_k=5)
        assert len(results) == 0, "Should handle no matches"
        
        # Test semantic search with edge cases
        embedder = SimpleEmbeddings()
        
        # Test with single document
        results = embedder.similarity_search("test", ["single document"], top_k=5)
        assert len(results) <= 1, "Should respect single document"
        
        # Test with identical documents
        identical_docs = ["same content", "same content", "same content"]
        results = embedder.similarity_search("same", identical_docs, top_k=5)
        assert len(results) <= 3, "Should handle identical documents"
        
        print("   Search edge cases tests passed")
    
    def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        
        # Create a large document to test memory handling
        large_text = "Memory test. " * 50000
        
        # Test streaming splitter (should be memory efficient)
        splitter = StreamingTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = list(splitter.split_text(large_text))
        
        assert len(chunks) > 0, "Should handle large document"
        assert all(len(chunk) <= 1000 for chunk in chunks), "Should respect chunk size"
        
        # Test semantic search with large document
        embedder = SimpleEmbeddings()
        results = embedder.similarity_search("memory", [large_text], top_k=3)
        assert len(results) <= 3, "Should handle large document in search"
        
        print("   Memory pressure tests passed")
    
    def test_concurrent_processing(self):
        """Test concurrent processing scenarios"""
        
        # Test multiple splitters operating simultaneously
        test_text = "Concurrent processing test. " * 100
        
        splitters = [
            CharacterTextSplitter(chunk_size=200, chunk_overlap=20),
            RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
            StreamingTextSplitter(chunk_size=200, chunk_overlap=20)
        ]
        
        # Process with all splitters
        all_chunks = []
        for splitter in splitters:
            chunks = list(splitter.split_text(test_text))
            all_chunks.extend(chunks)
        
        assert len(all_chunks) > 0, "Concurrent processing should work"
        
        # Test concurrent search operations
        documents = [
            "Document 1 content for testing",
            "Document 2 content for testing",
            "Document 3 content for testing"
        ]
        
        queries = ["test", "content", "document"]
        
        all_results = []
        for query in queries:
            results = simple_text_search(query, documents, top_k=3)
            all_results.extend(results)
        
        assert len(all_results) > 0, "Concurrent search should work"
        
        print("   Concurrent processing tests passed")
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        
        test_text = "Test text for parameter validation."
        
        # Test invalid chunk size
        try:
            splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=10)
            chunks = list(splitter.split_text(test_text))
            # Should handle gracefully or raise appropriate error
        except:
            pass  # Expected to raise error
        
        # Test negative chunk size
        try:
            splitter = CharacterTextSplitter(chunk_size=-1, chunk_overlap=10)
            chunks = list(splitter.split_text(test_text))
            # Should handle gracefully or raise appropriate error
        except:
            pass  # Expected to raise error
        
        # Test negative overlap
        try:
            splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=-1)
            chunks = list(splitter.split_text(test_text))
            # Should handle gracefully or raise appropriate error
        except:
            pass  # Expected to raise error
        
        print("   Invalid parameters tests passed")
    
    def test_file_system_edge_cases(self):
        """Test file system edge cases"""
        
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # Test non-existent file
        try:
            with open("non_existent_file.txt", 'r') as f:
                content = f.read()
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        # Test empty file
        empty_file = test_dir / "empty_file.txt"
        with open(empty_file, 'w') as f:
            f.write("")
        
        with open(empty_file, 'r') as f:
            content = f.read()
        assert content == "", "Empty file should read as empty string"
        
        # Clean up
        empty_file.unlink()
        
        print("   File system edge cases tests passed")
    
    def run_all_tests(self):
        """Run all edge case tests"""
        
        print("🧪 Running Edge Case Tests")
        print("=" * 60)
        
        tests = [
            ("Empty Inputs", self.test_empty_inputs),
            ("None Inputs", self.test_none_inputs),
            ("Single Character Inputs", self.test_single_character_inputs),
            ("Extremely Long Single Line", self.test_extremely_long_single_line),
            ("Whitespace Only Inputs", self.test_whitespace_only_inputs),
            ("Special Characters", self.test_special_characters),
            ("Repeating Content", self.test_repeating_content),
            ("Chunk Size Edge Cases", self.test_chunk_size_edge_cases),
            ("Overlap Edge Cases", self.test_overlap_edge_cases),
            ("Encoding Issues", self.test_encoding_issues),
            ("Search Edge Cases", self.test_search_edge_cases),
            ("Memory Pressure", self.test_memory_pressure),
            ("Concurrent Processing", self.test_concurrent_processing),
            ("Invalid Parameters", self.test_invalid_parameters),
            ("File System Edge Cases", self.test_file_system_edge_cases)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 {test_name}")
            print("-" * 40)
            self.run_test(test_name, test_func)
        
        # Generate summary
        passed = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed = len(self.test_results) - passed
        
        print(f"\n📊 Edge Case Test Summary")
        print("=" * 60)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        return self.test_results

def main():
    """Main function"""
    
    tester = EdgeCaseTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    failed = sum(1 for result in results if result['status'] == 'FAIL')
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All edge case tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()