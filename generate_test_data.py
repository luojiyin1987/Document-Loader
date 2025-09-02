#!/usr/bin/env python3
"""
Test data generator for document loader project
Creates various test files and scenarios for comprehensive testing
"""

import os
import json
import random
from pathlib import Path

class TestDataGenerator:
    """Generate comprehensive test data for document loader testing"""
    
    def __init__(self, output_dir="test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_basic_documents(self):
        """Generate basic test documents"""
        
        docs = {
            'simple.txt': """This is a simple test document.
It contains multiple lines of text.
Each line should be handled properly by the text splitter.
The document is short but suitable for basic testing.""",
            
            'medium.txt': """This is a medium-length test document that contains more content for testing purposes. 

The document includes multiple paragraphs with different themes and topics. Each paragraph should be processed independently by the text splitting algorithms.

Some paragraphs are longer than others, which helps test the chunk size limits and overlap functionality. The text splitter should be able to handle these variations gracefully.

The final paragraph contains conclusion text and summary information that should be preserved in the splitting process.""",
            
            'long.txt': """This is a long document designed to test the performance and memory efficiency of the text splitters. """ + \
            """This sentence is repeated many times to create a substantial amount of text. """ * 200 + \
            """The document ends here with some final text to ensure proper processing.""",
        }
        
        for filename, content in docs.items():
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return docs
    
    def generate_multilingual_documents(self):
        """Generate multilingual test documents"""
        
        docs = {
            'chinese.txt': """这是一个中文测试文档，用于测试中文文本处理功能。

第一段：中文文本包含多种字符和标点符号，需要正确的分词和处理。文本分割器应该能够正确处理中文段落和句子。

第二段：中文文本通常没有明显的空格分隔，这对文本分割提出了特殊的挑战。分割器需要能够理解中文的语义结构。

第三段：This paragraph contains mixed Chinese and English text to test mixed language processing capabilities. 混合语言处理是文档加载器的重要功能。

第四段：中文技术文档常常包含专业术语和英文缩写，如AI、ML、NLP等。这些应该被正确识别和处理。

第五段：总结来说，中文文本处理需要特别注意字符编码、分词策略和语义理解。""",
            
            'japanese.txt': """これは日本語のテスト文書です。

第一段落：日本語のテキスト処理機能をテストするための文書です。日本語にはひらがな、カタカナ、漢字など様々な文字が含まれています。

第二段落：テキストスプリッターは日本語の文章構造を正しく処理できる必要があります。句読点や改行を適切に扱うことが重要です。

第三段落：This is a mixed Japanese and English paragraph. 混合言語の処理もテストする必要があります。

第四段落：日本語の技術文書には多くの専門用語や外来語が含まれています。これらの正しい処理が求められます。""",
            
            'korean.txt': """이것은 한국어 테스트 문서입니다.

첫 번째 단락: 한국어 텍스트 처리 기능을 테스트하기 위한 문서입니다. 한국어는 한글과 한자를 포함한 복잡한 문자 시스템을 가지고 있습니다.

두 번째 단락: 텍스트 분할기는 한국어의 문장 구조를 올바르게 처리할 수 있어야 합니다. 적절한 단어 분리와 의미 파악이 중요합니다.

세 번째 단락: This is a mixed Korean and English paragraph. 다국어 처리 능력도 테스트해야 합니다.

네 번째 단락: 한국어 기술 문서에는 많은 전문 용어와 외래어가 포함되어 있습니다. 이들의 올바른 처리가 필요합니다.""",
            
            'multilingual.txt': """This document contains multiple languages:

English: This is the English portion of the document. It contains standard English text with proper grammar and structure.

中文：这是中文部分，包含标准的中文文本和标点符号。中文文本的处理需要特殊的分词策略。

日本語：これは日本語の部分です。日本語のテキスト処理にはひらがな、カタカナ、漢字の適切な扱いが必要です。

한국어: 이것은 한국어 부분입니다. 한국어 텍스트 처리에는 한글과 한자의 올바른 처리가 필요합니다.

Mixed: Sometimes sentences mix multiple languages like this English-Chinese mix: 技术文档通常包含English technical terms和中文解释。

The document demonstrates the ability to handle various languages and mixed-language scenarios effectively."""
        }
        
        for filename, content in docs.items():
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return docs
    
    def generate_technical_documents(self):
        """Generate technical and code-related documents"""
        
        docs = {
            'python_code.txt': '''#!/usr/bin/env python3
"""
Python code example for testing code processing capabilities
"""

import os
import sys
from typing import List, Dict, Any

class DataProcessor:
    """Data processing class with various methods"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data = []
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = data
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {file_path}")
            return []
    
    def process_data(self) -> Dict[str, Any]:
        """Process loaded data"""
        if not self.data:
            return {"error": "No data loaded"}
        
        processed = {
            "count": len(self.data),
            "processed_items": []
        }
        
        for item in self.data:
            processed_item = {
                "id": item.get("id"),
                "name": item.get("name", "").upper(),
                "value": item.get("value", 0) * 2
            }
            processed["processed_items"].append(processed_item)
        
        return processed
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save processing results"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

def main():
    """Main function"""
    processor = DataProcessor()
    data = processor.load_data("input.json")
    results = processor.process_data()
    processor.save_results(results, "output.json")

if __name__ == "__main__":
    main()''',
            
            'javascript_code.txt': '''/**
 * JavaScript code example for testing
 */

class DataProcessor {
    constructor(config = {}) {
        this.config = config;
        this.data = [];
    }
    
    /**
     * Load data from file or API
     */
    async loadData(source) {
        try {
            const response = await fetch(source);
            this.data = await response.json();
            return this.data;
        } catch (error) {
            console.error('Error loading data:', error);
            return [];
        }
    }
    
    /**
     * Process the loaded data
     */
    processData() {
        if (!this.data || this.data.length === 0) {
            return { error: "No data loaded" };
        }
        
        const processed = {
            count: this.data.length,
            processedItems: this.data.map(item => ({
                id: item.id,
                name: item.name?.toUpperCase() || '',
                value: (item.value || 0) * 2,
                timestamp: new Date().toISOString()
            }))
        };
        
        return processed;
    }
    
    /**
     * Save results to localStorage or download
     */
    saveResults(results, filename = 'results.json') {
        const jsonString = JSON.stringify(results, null, 2);
        
        // Save to localStorage
        localStorage.setItem('processedData', jsonString);
        
        // Download as file
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Usage example
const processor = new DataProcessor();
processor.loadData('https://api.example.com/data')
    .then(data => {
        const results = processor.processData();
        processor.saveResults(results);
    })
    .catch(error => {
        console.error('Processing failed:', error);
    });''',
            
            'technical_documentation.txt': '''# Technical Documentation: Text Processing System

## Overview
This document describes the text processing system architecture and implementation details. The system is designed to handle various types of documents and provide efficient text processing capabilities.

## Core Components

### 1. Text Splitter Module
The text splitter module is responsible for breaking down large documents into manageable chunks. It implements several splitting strategies:

- **Character-based splitting**: Splits text based on character count
- **Recursive splitting**: Uses hierarchical approach with separators
- **Token-based splitting**: Splits based on token count rather than characters
- **Semantic splitting**: Uses natural language processing for intelligent splitting
- **Streaming splitting**: Processes text in a memory-efficient streaming manner

### 2. Search Module
The search module provides multiple search strategies:

- **Keyword search**: Simple text matching based on keywords
- **Semantic search**: Uses embeddings to find semantically similar content
- **Hybrid search**: Combines keyword and semantic approaches for better results

### 3. Document Loader
The document loader supports multiple file formats:

- **Text files (.txt)**: Basic text file support
- **PDF files (.pdf)**: PDF document extraction
- **Web pages**: URL-based content loading

## Configuration Options

### Text Splitter Configuration
```python
splitter = TextSplitter(
    chunk_size=1000,      # Maximum characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=None,      # Custom separators
    keep_separator=True   # Whether to keep separators in chunks
)
```

### Search Configuration
```python
search = HybridSearch(
    keyword_weight=0.4,    # Weight for keyword matching
    semantic_weight=0.6    # Weight for semantic similarity
)
```

## Performance Considerations

### Memory Usage
- Use streaming splitter for large documents
- Monitor memory usage during processing
- Implement proper cleanup of temporary data

### Processing Speed
- Semantic search is slower but more accurate
- Keyword search is fast but less precise
- Hybrid search provides good balance

## Error Handling

The system implements comprehensive error handling:

- Graceful degradation from semantic to keyword search
- Proper handling of malformed documents
- Memory management for large files
- Encoding detection and conversion

## Best Practices

1. **Choose appropriate chunk size**: Balance between context preservation and memory usage
2. **Use semantic search for concept matching**: When looking for related concepts rather than exact matches
3. **Implement proper error handling**: Always handle exceptions and provide meaningful error messages
4. **Test with various document types**: Ensure compatibility with different formats and languages
5. **Monitor performance**: Regular testing and optimization of processing speed

## Future Enhancements

- Support for more file formats (DOCX, XLSX, etc.)
- Advanced NLP capabilities
- Distributed processing for large-scale operations
- Machine learning model integration
- Real-time processing capabilities

## Conclusion

This text processing system provides a robust foundation for document analysis and search operations. It balances performance, accuracy, and memory efficiency to handle various text processing scenarios effectively.''',
        }
        
        for filename, content in docs.items():
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return docs
    
    def generate_edge_case_documents(self):
        """Generate documents that test edge cases"""
        
        docs = {
            'empty.txt': "",
            
            'whitespace.txt': "    \n\n   \t\n   \n\n   ",
            
            'single_character.txt': "A",
            
            'single_line_long.txt': "This is a very long single line of text without any paragraph breaks or line endings to test how the text splitter handles extremely long lines that exceed the chunk size limit and need to be split appropriately without losing context or meaning " * 100,
            
            'repeating_content.txt': "Repeat. " * 1000,
            
            'special_characters.txt': """Special characters test:
!@#$%^&*()_+-={}[]|\\:;\"'<>?,./
~`•√π÷×¶§∆¢£¥€©®™✓❤️🔥🚀

Unicode characters:
中文العربيةहिन्दीРусский日本語한국어
éèêëāēîïôùûüçñß

Mathematical symbols:
∑∏∫√∞≈≠≤≥∈∉⊂⊃⊆⊇∪∩
αβγδεζηθικλμνξοπρστυφχψω
""",
            
            'encoding_test.txt': "Encoding test: Café, naïve, résumé, façade, Müller, Björn, Zoë, ½, ⅓, ¼, †, ‡, …, —, –",
            
            'minimal_content.txt': "A\nB\nC",
        }
        
        for filename, content in docs.items():
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return docs
    
    def generate_search_test_documents(self):
        """Generate documents specifically for search testing"""
        
        topics = [
            ("Python Programming", [
                "Python is a high-level programming language known for its simplicity and readability.",
                "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "Popular Python frameworks include Django, Flask, and FastAPI for web development.",
                "Python is widely used in data science, machine learning, and artificial intelligence applications.",
                "The Python Package Index (PyPI) hosts thousands of third-party libraries and packages."
            ]),
            ("Machine Learning", [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Supervised learning uses labeled data to train models for classification and regression tasks.",
                "Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.",
                "Reinforcement learning trains agents to make decisions through reward-based learning.",
                "Deep learning uses neural networks with multiple layers to extract hierarchical features."
            ]),
            ("Web Development", [
                "Frontend development involves creating user interfaces with HTML, CSS, and JavaScript.",
                "Backend development handles server-side logic, databases, and API integrations.",
                "Full-stack developers work on both frontend and backend components.",
                "Modern web frameworks include React, Vue.js, and Angular for frontend development.",
                "Backend technologies include Node.js, Django, Ruby on Rails, and ASP.NET."
            ]),
            ("Data Science", [
                "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "Data cleaning and preprocessing are crucial steps in the data science workflow.",
                "Exploratory data analysis helps understand patterns and relationships in datasets.",
                "Statistical analysis and hypothesis testing validate findings and support decision-making.",
                "Data visualization communicates insights through charts, graphs, and interactive dashboards."
            ]),
            ("Cybersecurity", [
                "Cybersecurity protects digital systems and data from unauthorized access and attacks.",
                "Network security monitors and protects network infrastructure from threats.",
                "Application security focuses on securing software applications throughout their lifecycle.",
                "Information security protects sensitive data from unauthorized access and disclosure.",
                "Security awareness training helps employees recognize and respond to security threats."
            ])
        ]
        
        # Create individual topic files
        topic_docs = {}
        for topic, sentences in topics:
            filename = topic.lower().replace(" ", "_") + ".txt"
            content = "\n".join(sentences)
            topic_docs[filename] = content
            
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create combined document
        combined_content = ""
        for topic, sentences in topics:
            combined_content += f"# {topic}\n\n"
            combined_content += "\n".join(sentences) + "\n\n"
        
        topic_docs['all_topics.txt'] = combined_content
        with open(self.output_dir / 'all_topics.txt', 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        return topic_docs
    
    def generate_large_test_document(self):
        """Generate a large document for performance testing"""
        
        # Generate a large document with structured content
        sections = [
            "Introduction",
            "Background",
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion",
            "References"
        ]
        
        content = ""
        for section in sections:
            content += f"# {section}\n\n"
            
            # Add multiple paragraphs per section
            for para in range(5):
                sentences = [
                    f"This is paragraph {para + 1} of the {section.lower()} section.",
                    "It contains multiple sentences that discuss various aspects of the topic.",
                    "The content is designed to test text splitting and search functionality.",
                    "Each paragraph contributes to the overall structure and coherence of the document.",
                    "The text should be processed efficiently by the document loading system."
                ]
                content += " ".join(sentences) + "\n\n"
        
        # Add technical content
        content += "# Technical Appendix\n\n"
        for i in range(50):
            content += f"Technical note {i + 1}: This section contains detailed technical information "
            content += f"about implementation details, algorithms, and system architecture. "
            content += f"The content is repeated to create a substantial document for testing purposes.\n\n"
        
        # Save the large document
        with open(self.output_dir / 'large_document.txt', 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {'large_document.txt': content}
    
    def generate_all(self):
        """Generate all test documents"""
        
        print("📁 Generating test documents...")
        
        all_docs = {}
        
        # Generate all document types
        generators = [
            ("Basic Documents", self.generate_basic_documents),
            ("Multilingual Documents", self.generate_multilingual_documents),
            ("Technical Documents", self.generate_technical_documents),
            ("Edge Case Documents", self.generate_edge_case_documents),
            ("Search Test Documents", self.generate_search_test_documents),
            ("Large Document", self.generate_large_test_document)
        ]
        
        for category, generator in generators:
            print(f"   Generating {category}...")
            docs = generator()
            all_docs.update(docs)
            print(f"   ✅ Generated {len(docs)} {category.lower()}")
        
        # Generate index file
        index_content = "# Test Documents Index\n\n"
        index_content += f"Generated {len(all_docs)} test documents for comprehensive testing.\n\n"
        index_content += "## Document Categories\n\n"
        
        categories = {
            "Basic Documents": [k for k in all_docs.keys() if k in ['simple.txt', 'medium.txt', 'long.txt']],
            "Multilingual Documents": [k for k in all_docs.keys() if k in ['chinese.txt', 'japanese.txt', 'korean.txt', 'multilingual.txt']],
            "Technical Documents": [k for k in all_docs.keys() if k in ['python_code.txt', 'javascript_code.txt', 'technical_documentation.txt']],
            "Edge Case Documents": [k for k in all_docs.keys() if k in ['empty.txt', 'whitespace.txt', 'single_character.txt', 'single_line_long.txt', 'repeating_content.txt', 'special_characters.txt', 'encoding_test.txt', 'minimal_content.txt']],
            "Search Test Documents": [k for k in all_docs.keys() if k in ['python_programming.txt', 'machine_learning.txt', 'web_development.txt', 'data_science.txt', 'cybersecurity.txt', 'all_topics.txt']],
            "Large Documents": [k for k in all_docs.keys() if k in ['large_document.txt']]
        }
        
        for category, files in categories.items():
            index_content += f"### {category}\n"
            for filename in files:
                if filename in all_docs:
                    size = len(all_docs[filename])
                    index_content += f"- {filename} ({size} characters)\n"
            index_content += "\n"
        
        with open(self.output_dir / 'index.md', 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        print(f"✅ Generated {len(all_docs)} test documents")
        print(f"📁 Documents saved to: {self.output_dir}")
        print(f"📋 Index file: {self.output_dir}/index.md")
        
        return all_docs

def main():
    """Main function to generate test data"""
    
    generator = TestDataGenerator()
    all_docs = generator.generate_all()
    
    print(f"\n🎉 Test data generation completed!")
    print(f"📊 Total documents: {len(all_docs)}")
    print(f"📁 Output directory: test_data/")

if __name__ == "__main__":
    main()